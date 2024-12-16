default_sample_rate = 44100
import librosa
import os
import numpy as np
np.float = float  # 🐒
import soundfile as sf
from scipy.signal import resample, find_peaks
from scipy.io import wavfile
from pathlib import Path
from music21 import converter
import pretty_midi as pm
from librosa import load, get_samplerate, pitch_tuning, hz_to_midi, time_to_samples, onset, stft
from scipy.signal import find_peaks, hilbert, peak_widths, butter, filtfilt, resample
import matplotlib.pyplot as plt
import crepe
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import madmom
from madmom.features import CNNOnsetProcessor
_original_CNNOnsetProcessor = CNNOnsetProcessor
from tqdm import tqdm
import json

#Preprocess functions

# Save the original CNNOnsetProcessor
_original_CNNOnsetProcessor = CNNOnsetProcessor

# Create a patched version
class PatchedCNNOnsetProcessor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        print("Patched CNNOnsetProcessor initialized.")

    def __call__(self, audio_path, *args, **kwargs):
        print("Patched CNNOnsetProcessor is being called.")
        # Use the detect_onsets function instead of direct processing
        return detect_onsets(audio_path, save_onsets=True)

madmom.features.CNNOnsetProcessor = PatchedCNNOnsetProcessor


# Ensure detect_onsets uses the original CNNOnsetProcessor
def detect_onsets(audio_path, save_onsets=True):
    """
    Detects onsets in the given audio file, with error handling.

    Args:
        audio_path (str or Path): Path to the audio file.
        save_onsets (bool): Whether to save detected onsets for later use.

    Returns:
        np.ndarray: Array of detected onset activations.
    """
    try:
        onsets_path = Path(audio_path).with_suffix(".onsets.npz")
        if onsets_path.exists():
            print(f"Loading onsets from {onsets_path}")
            onset_activations = np.load(onsets_path, allow_pickle=True)["activations"]
        else:
            print(f"Onsets file not found at {onsets_path}")
            print("Running onset detection...")
            onset_processor = _original_CNNOnsetProcessor()
            onset_activations = onset_processor(str(audio_path))
            if save_onsets:
                np.savez(onsets_path, activations=onset_activations)
    except Exception as e:
        raise RuntimeError(f"Error during onset detection: {e}")

    if onset_activations is None or len(onset_activations) == 0:
        raise ValueError("Onset detection failed: no activations detected.")

    return onset_activations



def musicxml_to_midi(musicxml_file, output_midi_file):
    try:
        score = converter.parse(musicxml_file)
        score.write('midi', fp=output_midi_file)
        print(f"MusicXML saved as MIDI: {output_midi_file}")
    except Exception as e:
        raise RuntimeError(f"Error converting MusicXML to MIDI: {e}")
    return output_midi_file


def add_default_instrument(self, instrument_name="Acoustic Grand Piano"):
    """
    为 PrettyMIDI 动态添加默认乐器的方法。
    """
    if len(self.instruments) == 0:
        instrument = pm.Instrument(
            program=pm.instrument_name_to_program(instrument_name)
        )
        self.instruments.append(instrument)
    return self.instruments[0]

pm.PrettyMIDI.add_default_instrument = add_default_instrument



def preprocess_audio(input_file, output_file):
    try:
        y, sr = librosa.load(input_file, sr=None)
        sf.write(output_file, y, sr)
        print(f"Audio file preprocessed and saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error preprocessing audio file: {e}")


def run_crepe(audio_path):
    audio, sr = librosa.load(str(audio_path), sr=None)  # Set sr=None to preserve the original sample rate
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    
    return frequency, confidence


def steps_to_samples(step_val, sr, step_size=0.01):
    return int(step_val * (sr * step_size))


def samples_to_steps(sample_val, sr, step_size=0.01):
    return int(sample_val / (sr * step_size))


def freqs_to_midi(freqs, tuning_offset=0):
    return np.nan_to_num(hz_to_midi(freqs) - tuning_offset, neginf=0)


def calculate_tuning_offset(freqs):
    tuning_offset = pitch_tuning(freqs)
    print(f"Tuning offset: {tuning_offset * 100} cents")
    return tuning_offset


def parse_f0(f0_path):
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    return np.nan_to_num(data['frequency']), np.nan_to_num(data['confidence'])
    
def save_f0(f0_path, frequency, confidence):
    np.savetxt(f0_path, np.stack([np.linspace(0, 0.01 * len(frequency), len(frequency)).astype('float'), frequency.astype('float'), confidence.astype('float')], axis=1), fmt='%10.7f', delimiter=',', header='time,frequency,confidence', comments='')
    return

def load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, save_amp_envelope):
    if cached_amp_envelope_path.exists():
        # if we have a cached amplitude envelope, no need to load audio
        filtered_amp_envelope = np.load(cached_amp_envelope_path, allow_pickle=True)['filtered_amp_envelope']
        # sr = get_samplerate(audio_path)
        sr = default_sample_rate # this is mainly to make tests work without having to load audio
        y = None
    else:
        try:
            y, sr = load(str(audio_path), sr=None)
        except:
            print("Error loading audio file. Amplitudes will be set to 80")
            detect_amplitude = False
            y = None
            pass

        amp_envelope = np.abs(hilbert(y))

        scaled_amp_envelope = np.interp(amp_envelope, (amp_envelope.min(), amp_envelope.max()), (0, 1))
        # low pass filter the amplitude envelope
        b, a = butter(4, 50, 'low', fs=sr)
        filtered_amp_envelope = filtfilt(b, a, scaled_amp_envelope)[::(sr//100)]
    
    if save_amp_envelope:
        np.savez(cached_amp_envelope_path, filtered_amp_envelope=filtered_amp_envelope)
    
    return sr, y, filtered_amp_envelope, detect_amplitude    

#Change performance audio in to performance midi###############################

def process(freqs,
            conf,
            audio_path,
            output_label="transcription",
            sensitivity=0.001,
            use_smoothing=False,
            min_duration=0.03,
            min_velocity=6,
            disable_splitting=False,
            use_cwd=True,
            tuning_offset=False,
            detect_amplitude=True,
            save_amp_envelope=False,
            default_sample_rate=44100,
            save_analysis_files=False,):
    
    cached_amp_envelope_path = audio_path.with_suffix(".amp_envelope.npz")
    sr, y, filtered_amp_envelope, detect_amplitude = load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, (save_analysis_files or save_amp_envelope))


    if use_cwd:
        output_filename = audio_path.stem
    else:
        output_filename = str(audio_path.parent) + "/" + audio_path.stem


    if save_analysis_files:
        f0_path = audio_path.with_suffix(".f0.csv")
        if not f0_path.exists():
            save_f0(f0_path, freqs, conf)
            
    timed_output_notes = []

    if not disable_splitting:
        onsets_path = str(audio_path.with_suffix('.onsets.npz'))
        if not os.path.exists(onsets_path):
            print(f"Onsets file not found at {onsets_path}")
            print("Running onset detection...")
            
            from madmom.features import CNNOnsetProcessor
            
            onset_activations = CNNOnsetProcessor()(str(audio_path))
            if save_analysis_files:
                np.savez(onsets_path, activations=onset_activations)
        else:
            print(f"Loading onsets from {onsets_path}")
            onset_activations = np.load(onsets_path, allow_pickle=True)['activations']

        onsets = np.zeros_like(onset_activations)
        onsets[find_peaks(onset_activations, distance=4, height=0.8)[0]] = 1

    if tuning_offset == False:
        tuning_offset = calculate_tuning_offset(freqs)
    else:
        tuning_offset = tuning_offset / 100
        

    # get pitch gradient
    midi_pitch = freqs_to_midi(freqs, tuning_offset)
    pitch_changes = np.abs(np.gradient(midi_pitch))
    pitch_changes = np.interp(pitch_changes,
                              (pitch_changes.min(), pitch_changes.max()),
                              (0, 1))

    # get confidence peaks with peak widths (prominences)
    conf_peaks, conf_peak_properties = find_peaks(1 - conf,
                                                  distance=4,
                                                  prominence=sensitivity)

    # combine pitch changes and confidence peaks to get change point signal
    change_point_signal = (1 - conf) * pitch_changes
    change_point_signal = np.interp(
        change_point_signal,
        (change_point_signal.min(), change_point_signal.max()), (0, 1))
    peaks, peak_properties = find_peaks(change_point_signal,
                                        distance=4,
                                        prominence=sensitivity)
    _, _, transition_starts, transition_ends = peak_widths(change_point_signal, peaks, rel_height=0.5)
    transition_starts = list(map(int, np.round(transition_starts)))
    transition_ends = list(map(int, np.round(transition_ends)))

    # get candidate note regions - any point between two peaks in the change point signal
    transitions = [(s, f, 'transition') for (s, f) in zip(transition_starts, transition_ends)]
    note_starts = [0] + transition_ends
    note_ends = transition_starts + [len(change_point_signal) + 1]
    note_regions = [(s, f, 'note') for (s, f) in (zip(note_starts, note_ends))]

    if detect_amplitude:
        # take the amplitudes within 6 sigma of the mean
        # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
        amp_mean = np.mean(filtered_amp_envelope)
        amp_sd = np.std(filtered_amp_envelope)
        # filtered_amp_envelope = amp_envelope.copy()
        filtered_amp_envelope[filtered_amp_envelope > amp_mean + (6 * amp_sd)] = 0
        global_max_amp = max(filtered_amp_envelope)

    segment_list = []
    for a, b, label in sum(zip(note_regions, transitions), ()):
        if label == 'transition':
            continue

        # Handle an edge case where rounding could cause
        # an end index for a note to be before the start index
        if a > b:
            continue
        elif b - a <= 1:
            continue

        if detect_amplitude:
            max_amp = np.max(filtered_amp_envelope[a:b])
            scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))
        else:
            scaled_max_amp = 80

        segment_list.append({
            'pitch': np.round(np.median(midi_pitch[a:b])),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1 - conf[a], # TODO: make use of the dip in confidence as a measure of how strong an onset is
            'amplitude': scaled_max_amp,
            'start_idx': a,
            'finish_idx': b,
        })

    # segment list contains our candidate notes
    # now we iterate through them and merge if two adjacent segments have the same median pitch
    notes = []
    sub_list = []
    for a, b in zip(segment_list, segment_list[1:]):
        # TODO: make use of variance in segment to catch glissandi?
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a['pitch'] -
                  b['pitch']) > 0.5:  # or a['transition_strength'] > 0.4:
            sub_list.append(a)
            notes.append(sub_list)
            sub_list = []
        else:
            sub_list.append(a)

    # catch any segments at the end
    if len(sub_list) > 0:
        notes.append(sub_list)

    output_midi = pm.PrettyMIDI()
    instrument = pm.Instrument(
        program=pm.instrument_name_to_program('Acoustic Grand Piano'))

    velocities = []
    durations = []
    output_notes = []

    # Filter out notes that are too short or too quiet
    for x_s in notes:
        x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity]
        if len(x_s_filt) == 0:
            continue
        median_pitch = np.median(np.array([y['pitch'] for y in x_s_filt]))
        median_confidence = np.median(np.array([y['conf'] for y in x_s_filt]))
        seg_start = x_s_filt[0]['start_idx']
        seg_end = x_s_filt[-1]['finish_idx']
        time_start = 0.01 * seg_start
        time_end = 0.01 * seg_end
        sample_start = time_to_samples(time_start, sr=sr)
        sample_end = time_to_samples(time_end, sr=sr)
        max_amp = np.max(filtered_amp_envelope[seg_start:seg_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = scaled_max_amp > min_velocity
        valid_duration = (time_end - time_start) > min_duration
        
        # TODO: make use of confidence strength
        valid_confidence = True  # median_confidence > 0.1

        if valid_amplitude and valid_confidence and valid_duration:
            output_notes.append({
                'pitch':
                    int(np.round(median_pitch)),
                'velocity':
                    round(scaled_max_amp),
                'start_idx':
                    seg_start,
                'finish_idx':
                    seg_end,
                'conf':
                    median_confidence,
                'transition_strength':
                    x_s[-1]['transition_strength']
            })

    # Handle repeated notes
    # Here we use a standard onset detection algorithm from madmom
    # with a high threshold (0.8) to re-split notes that are repeated
    # Repeated notes have a pitch gradient of 0 and are therefore
    # not separated by the algorithm above
    if not disable_splitting:
        onset_separated_notes = []
        for n in output_notes:
            n_s = n['start_idx']
            n_f = n['finish_idx']

            last_onset = 0
            if np.any(onsets[n_s:n_f] > 0.7):
                onset_idxs_within_note = np.argwhere(onsets[n_s:n_f] > 0.7)
                for idx in onset_idxs_within_note:
                    if idx[0] > last_onset + int(min_duration / 0.01):
                        new_note = n.copy()
                        new_note['start_idx'] = n_s + last_onset
                        new_note['finish_idx'] = n_s + idx[0]
                        onset_separated_notes.append(new_note)
                        last_onset = idx[0]

            # If there are no valid onsets within the range
            # the following should append a copy of the original note,
            # but if there were splits at onsets then it will also clean up any tails
            # left in the sequence
            new_note = n.copy()
            new_note['start_idx'] = n_s + last_onset
            new_note['finish_idx'] = n_f
            onset_separated_notes.append(new_note)
            output_notes = onset_separated_notes

    if detect_amplitude:
        # Trim notes that fall below a certain amplitude threshold
        timed_output_notes = []
        for n in output_notes:
            timed_note = n.copy()

            # Adjusting the start time to meet a minimum amp threshold
            s = timed_note['start_idx']
            f = timed_note['finish_idx']

            if f - s > (min_duration / 0.01):
                # TODO: make noise floor configurable
                noise_floor = 0.01  # this will vary depending on the signal
                s_samp = steps_to_samples(s, sr)
                f_samp = steps_to_samples(f, sr)
                s_adj_samp_all = s_samp + np.where(
                    filtered_amp_envelope[s:f] > noise_floor)[0]

                if len(s_adj_samp_all) > 0:
                    s_adj_samp_idx = s_adj_samp_all[0]
                else:
                    continue

                s_adj = samples_to_steps(s_adj_samp_idx, sr)

                f_adj_samp_idx = f_samp - np.where(
                    np.flip(filtered_amp_envelope[s:f]) > noise_floor)[0][0]
                if f_adj_samp_idx > f_samp or f_adj_samp_idx < 1:
                    print("something has gone wrong")

                f_adj = samples_to_steps(f_adj_samp_idx, sr)
                if f_adj > f or f_adj < 1:
                    print("something has gone more wrong")

                timed_note['start'] = s_adj * 0.01
                timed_note['finish'] = f_adj * 0.01
                timed_output_notes.append(timed_note)
            else:
                timed_note['start'] = s * 0.01
                timed_note['finish'] = f * 0.01

    for n in timed_output_notes:
        if n['start'] >= n['finish']:
            continue

        instrument.notes.append(
            pm.Note(start=n['start'],
                    end=n['finish'],
                    pitch=n['pitch'],
                    velocity=n['velocity']))

    output_midi.instruments.append(instrument)
    output_midi.write(f'{output_filename}.{output_label}.mid')
    output_folder = os.path.dirname(output_filename)
    record_audio_midi_mapping(timed_output_notes, sr, output_folder, output_label="audio_midi_mapping")

    return f"{output_filename}.{output_label}.mid"

    
#Adjust functions#########################################



def record_audio_midi_mapping(output_notes, sr, output_folder, output_label="audio_midi_mapping"):
    """
    Records the relationship between MIDI notes and their corresponding audio segments.

    Args:
        output_notes (list): List of notes with timing and sample index information.
        sr (int): Sample rate of the audio.
        output_folder (str): Folder to save the mapping file.
        output_label (str): Label for the mapping file.

    Returns:
        str: Path to the saved mapping file.
    """
    import json
    import librosa

    mapping = []
    for note in output_notes:
        mapping.append({
            "pitch": note["pitch"],
            "velocity": note["velocity"],
            "start_time": note["start_idx"] * 0.01,
            "end_time": note["finish_idx"] * 0.01,
            "sample_start": int(librosa.time_to_samples(note["start_idx"] * 0.01, sr=sr)),
            "sample_end": int(librosa.time_to_samples(note["finish_idx"] * 0.01, sr=sr))
        })

    mapping_file_path = os.path.join(output_folder, f"{output_label}.json")
    with open(mapping_file_path, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"Audio-MIDI mapping saved to: {mapping_file_path}")
    return mapping_file_path



def process_with_adjustments(audio_path, reference_midi_path, output_folder):
    # Generate performance MIDI (existing logic)
    performance_midi_file = process(
        freqs=frequency,
        conf=confidence,
        audio_path=Path(audio_path),
        output_label="performance",
        default_sample_rate=44100
    )

    # Move the MIDI file to the output folder
    performance_midi_path = Path(performance_midi_file)
    if performance_midi_path.exists():
        final_midi_path = Path(output_folder) / performance_midi_path.name
        os.rename(performance_midi_path, final_midi_path)
        performance_midi_file = str(final_midi_path)

    # Correct performance audio
    output_audio_file = os.path.join(output_folder, "corrected_performance_audio.wav")
    correct_performance_audio(audio_path, performance_midi_file, reference_midi_path, output_audio_file)



def correct_performance_audio(audio_file, performance_midi_file, reference_midi_file, output_audio_file):
    """
    Matches notes between performance and reference MIDI, adjusts performance audio
    based on correction factors, and outputs the improved performance audio.
    """
    # Load performance and reference MIDI files
    perf_pm = pm.PrettyMIDI(performance_midi_file)
    ref_pm = pm.PrettyMIDI(reference_midi_file)

    # Extract notes (start_time, pitch, duration)
    performance_notes = [(note.start, note.pitch, note.end - note.start)
                         for inst in perf_pm.instruments for note in inst.notes]
    reference_notes = [(note.start, note.pitch, note.end - note.start)
                       for inst in ref_pm.instruments for note in inst.notes]

    # Match notes between performance and reference MIDI
    note_mappings = []
    for perf_note in performance_notes:
        perf_start, perf_pitch, perf_duration = perf_note
        closest_ref_note = min(
            reference_notes,
            key=lambda ref_note: abs(ref_note[0] - perf_start) + abs(ref_note[1] - perf_pitch)
        )
        ref_start, ref_pitch, ref_duration = closest_ref_note

        # Calculate correction factors
        time_correction = ref_duration / perf_duration if perf_duration > 0 else 1.0
        relative_offset = (ref_start - reference_notes[0][0]) - (perf_start - performance_notes[0][0])

        # Store the mapping and correction factors
        note_mappings.append({
            "performance_note": perf_note,
            "reference_note": closest_ref_note,
            "time_correction": time_correction,
            "relative_offset": relative_offset
        })

    # Load the performance audio
    y, sr = librosa.load(audio_file, sr=None)
    corrected_audio = np.zeros_like(y)

    # Correct each audio segment corresponding to a performance note
    for mapping in note_mappings:
        perf_note = mapping["performance_note"]
        time_correction = mapping["time_correction"]
        relative_offset = mapping["relative_offset"]

        # Locate the audio segment
        perf_start_idx = librosa.time_to_samples(perf_note[0], sr)
        perf_end_idx = librosa.time_to_samples(perf_note[0] + perf_note[2], sr)
        segment = y[perf_start_idx:perf_end_idx]

        # Adjust segment timing
        adjusted_segment = librosa.effects.time_stretch(segment, rate=1/time_correction)

        # Calculate start position for corrected audio (applying offset)
        corrected_start_idx = int(perf_start_idx + relative_offset * sr)
        corrected_end_idx = corrected_start_idx + len(adjusted_segment)

        # Add the adjusted segment to the corrected audio
        if corrected_end_idx <= len(corrected_audio):  # Ensure no overflow
            corrected_audio[corrected_start_idx:corrected_end_idx] += adjusted_segment
        else:
            corrected_audio[corrected_start_idx:] += adjusted_segment[:len(corrected_audio) - corrected_start_idx]

    # Normalize corrected audio to prevent clipping
    corrected_audio = librosa.util.normalize(corrected_audio)

    # Save the corrected audio file
    sf.write(output_audio_file, corrected_audio, sr)
    print(f"Corrected performance audio saved to: {output_audio_file}")



def calculate_relative_metrics(notes):

    if not notes:
        return []

    first_note_time = notes[0][0]
    first_note_duration = notes[0][2]

    relative_notes = []
    for i, (start_time, pitch, duration) in enumerate(notes):
        relative_time = (start_time - first_note_time) / first_note_duration
        relative_duration = duration / first_note_duration
        relative_notes.append({
            "index": i,
            "start_time": start_time,
            "pitch": pitch,
            "duration": duration,
            "relative_time": relative_time,
            "relative_duration": relative_duration
        })

    return relative_notes



from tqdm import tqdm

def find_matching_notes(performance_notes, reference_notes, bpm):
    """
    Matches performance MIDI notes to reference MIDI notes with improved tolerance,
    fallback, and unmatched note preservation.
    
    Args:
        performance_notes (list of tuples): Performance MIDI notes as (start_time, pitch, duration).
        reference_notes (list of tuples): Reference MIDI notes as (start_time, pitch, duration).
        bpm (float): Beats per minute of the reference MIDI.

    Returns:
        list of dict: Matching note pairs with time and offset correction factors, including retained unmatched notes.
    """
    perf_rel_notes = calculate_relative_metrics(performance_notes)
    ref_rel_notes = calculate_relative_metrics(reference_notes)

    matches = []
    unmatched_notes = []
    tolerance = max(0.1, 60 / (4 * bpm))  # 16th note tolerance with a minimum threshold

    # First pass: Match notes with relative metrics
    for perf_note in perf_rel_notes:
        closest_match = None
        min_distance = float('inf')

        for ref_note in ref_rel_notes:
            if ref_note["pitch"] == perf_note["pitch"]:
                time_diff = abs(ref_note["relative_time"] - perf_note["relative_time"])
                if time_diff <= tolerance and time_diff < min_distance:
                    closest_match = ref_note
                    min_distance = time_diff

        if closest_match:
            time_correction = closest_match["duration"] / perf_note["duration"] if perf_note["duration"] > 0 else 1.0
            relative_offset = closest_match["start_time"] - perf_note["start_time"]
            matches.append({
                "performance_note": perf_note,
                "reference_note": closest_match,
                "time_correction": time_correction,
                "relative_offset": relative_offset
            })
        else:
            unmatched_notes.append(perf_note)

    # Fallback: Match unmatched notes by pitch and duration
    if unmatched_notes:
        retained_unmatched_notes = []
        for perf_note in unmatched_notes:
            closest_match = min(
                ref_rel_notes,
                key=lambda ref_note: (
                    abs(ref_note["pitch"] - perf_note["pitch"]) +
                    abs(ref_note["duration"] - perf_note["duration"])
                ),
                default=None
            )

            # Check if this note conflicts with any matched notes
            if closest_match:
                preceding_match = next((m for m in matches if m["performance_note"]["start_time"] < perf_note["start_time"]), None)
                following_match = next((m for m in matches if m["performance_note"]["start_time"] > perf_note["start_time"]), None)

                no_conflict = True
                if preceding_match:
                    no_conflict &= perf_note["start_time"] + perf_note["duration"] <= preceding_match["performance_note"]["start_time"]
                if following_match:
                    no_conflict &= perf_note["start_time"] >= following_match["performance_note"]["start_time"] + following_match["performance_note"]["duration"]

                if no_conflict:
                    retained_unmatched_notes.append(perf_note)

        # Add retained unmatched notes to matches as-is
        for note in retained_unmatched_notes:
            matches.append({
                "performance_note": note,
                "reference_note": None,  # Unmatched notes won't have a reference
                "time_correction": 1.0,  # No adjustment
                "relative_offset": 0.0
            })

    if not matches:
        print("Warning: No matches found. Ensure MIDI files are aligned and tempos match.")
    return matches



def adjust_audio_segments(note_mappings, audio_file, output_folder):
    """
    Adjusts audio segments based on note mappings and outputs the corrected audio.

    Args:
        note_mappings (list): Mappings of performance and reference notes with corrections.
        audio_file (str): Path to the original performance audio file.
        output_folder (str): Directory to save the adjusted audio.
    """
    try:
        y, sr = sf.read(audio_file)
    except Exception as e:
        raise RuntimeError(f"Error reading audio file: {e}")

    if not note_mappings:
        raise ValueError("No valid note mappings found. Ensure MIDI alignment is correct.")

    adjusted_audio = np.zeros_like(y)
    any_segment_adjusted = False

    for i, mapping in enumerate(note_mappings):
        perf_note = mapping["performance_note"]
        orig_note = mapping["reference_note"]

        try:
            # Calculate correction factors
            time_correction = max(0.1, mapping["time_correction"])  # Avoid zero or very small values
            offset_correction = mapping["relative_offset"]

            # Extract and adjust audio segment
            start_idx = librosa.time_to_samples(perf_note["start_time"], sr=sr)
            end_idx = librosa.time_to_samples(perf_note["start_time"] + perf_note["duration"], sr=sr)
            segment = y[start_idx:end_idx]

            if len(segment) == 0:
                print(f"[Note {i}] Skipping empty segment: {perf_note}")
                continue

            # Apply time correction
            adjusted_segment = librosa.effects.time_stretch(segment, rate=1 / time_correction)

            # Place the adjusted segment in the output
            corrected_start_idx = int(start_idx + offset_correction * sr)
            corrected_end_idx = corrected_start_idx + len(adjusted_segment)

            if corrected_end_idx <= len(adjusted_audio):
                adjusted_audio[corrected_start_idx:corrected_end_idx] += adjusted_segment
                any_segment_adjusted = True
            else:
                adjusted_audio[corrected_start_idx:] += adjusted_segment[:len(adjusted_audio) - corrected_start_idx]
                any_segment_adjusted = True

        except Exception as e:
            print(f"[Note {i}] Error processing segment: {e}")
            continue

    if not any_segment_adjusted:
        raise ValueError("No segments were successfully adjusted. Check note mappings and audio alignment.")

    # Normalize audio
    adjusted_audio = librosa.util.normalize(adjusted_audio)

    # Save adjusted audio
    output_audio_file = os.path.join(output_folder, "adjusted_output.wav")
    try:
        sf.write(output_audio_file, adjusted_audio, sr)
        print(f"Adjusted audio saved to: {output_audio_file}")
    except Exception as e:
        raise RuntimeError(f"Error saving adjusted audio: {e}")



def show_gui_with_autofit(original_midi, performance_midi):
    """
    在 GUI 中显示自动匹配后的结果。

    Args:
        original_midi (str): 参考 MIDI 文件路径。
        performance_midi (str): 演奏 MIDI 文件路径。
    """
    # 加载 MIDI
    original_pm = pm.PrettyMIDI(original_midi)
    performance_pm = pm.PrettyMIDI(performance_midi)

    # 提取音符
    original_notes = [(note.start, note.pitch, note.end - note.start)
                      for inst in original_pm.instruments for note in inst.notes]
    performance_notes = [(note.start, note.pitch, note.end - note.start)
                         for inst in performance_pm.instruments for note in inst.notes]

    # 自动匹配音符
    matches = find_matching_notes(performance_notes, original_notes)
    print("自动匹配结果：", matches)

    # 调用原始 GUI 显示
    show_gui(original_midi, performance_midi, [])


#Main GUI#############################################


def select_file(entry_widget):
    file_path = filedialog.askopenfilename()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, file_path)

def select_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, folder_path)


def process_and_visualize(audio_file, musicxml_file, output_folder, match_mode="automatic"):
    """
    Main processing function that handles audio preprocessing, MIDI conversion, note matching,
    and audio segment adjustments. Outputs adjusted audio and logs note correspondences.

    Args:
        audio_file (str): Path to the input audio file.
        musicxml_file (str): Path to the MusicXML file.
        output_folder (str): Directory to save the output files.
        match_mode (str): Matching mode, either 'manual' or 'automatic'. Default is 'automatic'.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.isfile(audio_file) or not os.path.isfile(musicxml_file):
        print("请选择有效的文件路径！")
        return

    try:
        # Convert MusicXML to MIDI
        musicxml_midi_file = os.path.join(output_folder, "musicxml.mid")
        musicxml_to_midi(musicxml_file, musicxml_midi_file)

        # Preprocess audio and run CREPE analysis
        cleaned_audio_file = os.path.join(output_folder, "cleaned_audio.wav")
        preprocess_audio(audio_file, cleaned_audio_file)

        frequency, confidence = run_crepe(cleaned_audio_file)

        # Generate performance MIDI
        performance_midi_file = process(
            freqs=frequency,
            conf=confidence,
            audio_path=Path(cleaned_audio_file),
            output_label="performance",
            default_sample_rate=44100
        )

        # Move the output MIDI file to the output folder
        performance_midi_path = Path(performance_midi_file)
        if performance_midi_path.exists():
            final_midi_path = Path(output_folder) / performance_midi_path.name
            os.rename(performance_midi_path, final_midi_path)
            performance_midi_file = str(final_midi_path)

        # Match Notes
        original_pm = pm.PrettyMIDI(musicxml_midi_file)
        performance_pm = pm.PrettyMIDI(performance_midi_file)
        original_notes = [(note.start, note.pitch, note.end - note.start)
                          for inst in original_pm.instruments for note in inst.notes]
        performance_notes = [(note.start, note.pitch, note.end - note.start)
                             for inst in performance_pm.instruments for note in inst.notes]

        bpm = original_pm.estimate_tempo()
        matches = find_matching_notes(performance_notes, original_notes, bpm)

        # Adjust audio segments
        adjust_audio_segments(matches, cleaned_audio_file, output_folder)

        # Log correspondences
        correspondence_log = os.path.join(output_folder, "correspondence_log.json")
        with open(correspondence_log, "w") as log_file:
            json.dump(matches, log_file, indent=4)

        # Remove intermediate files
        for intermediate_file in [
            musicxml_midi_file,
            cleaned_audio_file,
            cleaned_audio_file.replace(".wav", ".onsets.npz"),
        ]:
            if os.path.exists(intermediate_file):
                os.remove(intermediate_file)

    except Exception as e:
        print(f"Error: {e}.")




# 主界面逻辑
def start_gui():
    root = tk.Tk()
    root.title("音频和 MIDI 处理工具")

    frame = ttk.Frame(root)
    frame.pack(pady=10, padx=10)

    # File selection
    audio_label = ttk.Label(frame, text="选择音频文件:")
    audio_label.grid(row=0, column=0, sticky="w")
    audio_entry = ttk.Entry(frame, width=50)
    audio_entry.grid(row=0, column=1, padx=5)
    audio_button = ttk.Button(frame, text="选择", command=lambda: select_file(audio_entry))
    audio_button.grid(row=0, column=2)

    musicxml_label = ttk.Label(frame, text="选择 MusicXML 文件:")
    musicxml_label.grid(row=1, column=0, sticky="w")
    musicxml_entry = ttk.Entry(frame, width=50)
    musicxml_entry.grid(row=1, column=1, padx=5)
    musicxml_button = ttk.Button(frame, text="选择", command=lambda: select_file(musicxml_entry))
    musicxml_button.grid(row=1, column=2)

    output_label = ttk.Label(frame, text="选择输出文件夹:")
    output_label.grid(row=2, column=0, sticky="w")
    output_entry = ttk.Entry(frame, width=50)
    output_entry.grid(row=2, column=1, padx=5)
    output_button = ttk.Button(frame, text="选择", command=lambda: select_folder(output_entry))
    output_button.grid(row=2, column=2)


    # Process button
    process_button = ttk.Button(
        frame,
        text="处理并显示 MIDI",
        command=lambda: process_and_visualize(
            audio_entry.get(), musicxml_entry.get(), output_entry.get()
        )
    )
    process_button.grid(row=4, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    start_gui()
