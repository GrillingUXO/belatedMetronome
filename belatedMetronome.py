default_sample_rate = 44100
import librosa
import os
import numpy as np
np.float = float  # 🐒
import soundfile as sf
from scipy.io import wavfile
from pathlib import Path
from music21 import converter
import pretty_midi as pm
from librosa import load, get_samplerate, pitch_tuning, hz_to_midi, time_to_samples, onset, stft
from scipy.signal import find_peaks, hilbert, peak_widths, butter, filtfilt, resample
import matplotlib.pyplot as plt
import crepe
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import madmom
from madmom.features import CNNOnsetProcessor
_original_CNNOnsetProcessor = CNNOnsetProcessor
from tqdm import tqdm
import json
import moviepy
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import ffmpeg
from moviepy.video.fx import all as vfx
from moviepy.video.fx.speedx import speedx
from scipy.ndimage import gaussian_filter1d


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




def preprocess_audio(input_video_file, output_audio_file):
    """
    Extracts audio from a video file and preprocesses it for further analysis.
    
    Args:
        input_video_file (str): Path to the input video file.
        output_audio_file (str): Path to save the extracted audio file.
    """
    try:
        video_clip = VideoFileClip(input_video_file)
        audio = video_clip.audio
        audio.write_audiofile(output_audio_file, fps=44100)
        print(f"Audio extracted and saved to {output_audio_file}")
    except Exception as e:
        raise RuntimeError(f"Error extracting audio from video file: {e}")


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
            sensitivity=0.0005,
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
    raw_pitch_changes = np.abs(np.gradient(midi_pitch))
    smoothed_pitch_changes = gaussian_filter1d(raw_pitch_changes, sigma=1)
    pitch_changes = np.where(raw_pitch_changes > 0.5, raw_pitch_changes, smoothed_pitch_changes)



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
    _, _, transition_starts, transition_ends = peak_widths(change_point_signal, peaks, rel_height=0.7)
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
        mapping_entry = {
            "pitch": note["pitch"],
            "velocity": note["velocity"],
            "start_time": note["start_idx"] * 0.01,
            "end_time": note["finish_idx"] * 0.01,
            "sample_start": int(librosa.time_to_samples(note["start_idx"] * 0.01, sr=sr)),
            "sample_end": int(librosa.time_to_samples(note["finish_idx"] * 0.01, sr=sr))
        }
        if "reference_note" in note and note["reference_note"] is None:
            mapping_entry["status"] = "unmatched"
        else:
            mapping_entry["status"] = "matched"

        mapping.append(mapping_entry)

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

    # Adjust audio based on note mappings
    note_mappings = find_matching_notes(performance_notes, reference_notes, bpm)
    adjust_audio_segments(note_mappings, audio_path, video_file, output_folder)


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def find_matching_notes(performance_notes, reference_notes, bpm):
    if bpm <= 0:
        raise ValueError("BPM must be greater than 0.")

    seconds_per_beat = 60 / bpm

    def calculate_relative_metrics(notes):
        """Calculate relative duration and relative position for each note."""
        relative_notes = []
        cumulative_duration = 0
        for i, (start_time, pitch, duration) in enumerate(notes):
            relative_duration = duration / seconds_per_beat
            relative_position = cumulative_duration
            cumulative_duration += relative_duration
            relative_notes.append((i, pitch, relative_duration, relative_position))
        return relative_notes

    perf_rel_notes = calculate_relative_metrics(performance_notes)
    ref_rel_notes = calculate_relative_metrics(reference_notes)

    matches = []
    unmatched_perf_indices = set(range(len(performance_notes)))
    unmatched_ref_indices = set(range(len(reference_notes)))
    position_tolerance = 3  # Allowable relative position difference in beats

    def dtw_context_similarity(perf_idx, ref_idx):
        """Compute DTW similarity score for note contexts."""
        perf_context = perf_rel_notes[max(0, perf_idx - 1): perf_idx + 2]
        ref_context = ref_rel_notes[max(0, ref_idx - 1): ref_idx + 2]
        distance, _ = fastdtw(perf_context, ref_context, dist=euclidean)
        return distance

    # First matching round
    for perf_idx, (perf_order, perf_pitch, perf_duration, perf_position) in enumerate(perf_rel_notes):
        best_match = None
        best_score = float('inf')

        for ref_idx, (ref_order, ref_pitch, ref_duration, ref_position) in enumerate(ref_rel_notes):
            if ref_idx not in unmatched_ref_indices:
                continue
            if ref_pitch != perf_pitch:
                continue

            # Relative duration matching
            duration_diff = abs(ref_duration - perf_duration)
            duration_tolerance = 0.5 * ref_duration  # 0.5 times the reference note's relative duration
            if duration_diff > duration_tolerance:
                continue  # Skip if relative duration mismatch exceeds tolerance

            # Relative position matching (only checked if duration match passes)
            position_diff = abs(ref_position - perf_position)
            if position_diff > position_tolerance:
                continue

            # Calculate DTW context similarity (optional for fine-tuning match score)
            context_score = dtw_context_similarity(perf_idx, ref_idx)

            # Prioritize duration and position match over context similarity
            if context_score < best_score:
                best_match = (ref_idx, ref_pitch, ref_duration, ref_position, context_score)
                best_score = context_score

        if best_match:
            ref_idx, ref_pitch, ref_duration, ref_position, dtw_score = best_match
            relative_offset = ref_position - perf_position
            time_correction = ref_duration / perf_duration if perf_duration > 0 else 1.0

            matches.append({
                "order": perf_order,
                "performance_note": {
                    "start": performance_notes[perf_idx][0],
                    "pitch": performance_notes[perf_idx][1],
                    "duration": performance_notes[perf_idx][2]
                },
                "reference_note": {
                    "start": reference_notes[ref_idx][0],
                    "pitch": reference_notes[ref_idx][1],
                    "duration": reference_notes[ref_idx][2]
                },
                "original_relative_position": perf_position,
                "corrected_relative_position": ref_position,
                "time_correction": time_correction,
                "relative_offset": relative_offset,
                "match_round": 1,
                "dtw_score": dtw_score
            })
            unmatched_perf_indices.discard(perf_idx)
            unmatched_ref_indices.discard(ref_idx)

    # Second matching round for unmatched notes
    for perf_idx in list(unmatched_perf_indices):
        perf_order, perf_pitch, perf_duration, perf_position = perf_rel_notes[perf_idx]
        for ref_idx in list(unmatched_ref_indices):
            ref_order, ref_pitch, ref_duration, ref_position = ref_rel_notes[ref_idx]
            if perf_pitch == ref_pitch:
                context_score = dtw_context_similarity(perf_idx, ref_idx)  # For reference only
                time_correction = ref_duration / perf_duration if perf_duration > 0 else 1.0
                matches.append({
                    "order": perf_order,
                    "performance_note": {
                        "start": performance_notes[perf_idx][0],
                        "pitch": performance_notes[perf_idx][1],
                        "duration": performance_notes[perf_idx][2]
                    },
                    "reference_note": {
                        "start": reference_notes[ref_idx][0],
                        "pitch": reference_notes[ref_idx][1],
                        "duration": reference_notes[ref_idx][2]
                    },
                    "original_relative_position": perf_position,
                    "corrected_relative_position": ref_position,
                    "time_correction": time_correction,
                    "relative_offset": 0.0,
                    "match_round": 2,
                    "dtw_score": context_score
                })
                unmatched_perf_indices.discard(perf_idx)
                unmatched_ref_indices.discard(ref_idx)
                break

    # Handle unmatched performance notes
    for perf_idx in unmatched_perf_indices:
        perf_order, perf_pitch, perf_duration, perf_position = perf_rel_notes[perf_idx]
        perf_note = performance_notes[perf_idx]
        matches.append({
            "order": perf_order,
            "performance_note": {
                "start": perf_note[0],
                "pitch": perf_note[1],
                "duration": perf_note[2]
            },
            "reference_note": None,
            "original_relative_position": perf_position,
            "corrected_relative_position": perf_position,
            "time_correction": 1.0,
            "relative_offset": 0.0,
            "match_round": "Gap",
            "dtw_score": "N/A"
        })

    matches.sort(key=lambda match: match["order"])
    return matches



def convert_perf_note_types(note_mappings):
    """
    Converts the start time, pitch, and duration of performance notes to numeric types.

    Args:
        note_mappings (list): List of note mappings, each containing a "performance_note".

    Returns:
        list: Updated note_mappings with all performance note values converted to numeric types.
    """
    for i, mapping in enumerate(note_mappings):
        perf_note = mapping.get("performance_note")
        if isinstance(perf_note, dict):
            try:
                # Convert start_time, pitch, and duration to numeric types
                start_time = float(perf_note.get("start", 0))
                pitch = int(perf_note.get("pitch", 0))
                duration = float(perf_note.get("duration", 0))
                mapping["performance_note"] = {"start": start_time, "pitch": pitch, "duration": duration}
                print(f"[Note {i}] Converted performance_note: {mapping['performance_note']}")
            except (ValueError, TypeError) as e:
                print(f"[Note {i}] Error converting performance_note: {perf_note}, Error: {e}")
    return note_mappings



def find_nearest_zero_crossing(signal, target_idx):
    """
    Find the nearest zero-crossing index to the target index in a signal.
    Args:
        signal (np.ndarray): The 1D signal array to search in.
        target_idx (int): The index around which to search for a zero crossing.
    Returns:
        int: The index of the nearest zero crossing to the target.
    """
    if target_idx <= 0 or target_idx >= len(signal):
        return target_idx

    # Search backward and forward for zero crossings
    forward_idx = next((i for i in range(target_idx, len(signal)) if signal[i - 1] * signal[i] <= 0), len(signal) - 1)
    backward_idx = next((i for i in range(target_idx, 0, -1) if signal[i - 1] * signal[i] <= 0), 0)

    # Return the closer zero crossing
    if abs(forward_idx - target_idx) < abs(target_idx - backward_idx):
        return forward_idx
    else:
        return backward_idx


import numpy as np
import librosa
import soundfile as sf
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from moviepy.video.fx import speedx
import subprocess


def find_nearest_zero_crossing(signal, target_idx):
    """
    Find the nearest zero-crossing index to the target index in a signal.
    Args:
        signal (np.ndarray): The 1D signal array to search in.
        target_idx (int): The index around which to search for a zero crossing.
    Returns:
        int: The index of the nearest zero crossing to the target.
    """
    if target_idx <= 0 or target_idx >= len(signal):
        return target_idx

    # Search backward and forward for zero crossings
    forward_idx = next((i for i in range(target_idx, len(signal)) if signal[i - 1] * signal[i] <= 0), len(signal) - 1)
    backward_idx = next((i for i in range(target_idx, 0, -1) if signal[i - 1] * signal[i] <= 0), 0)

    # Return the closer zero crossing
    if abs(forward_idx - target_idx) < abs(target_idx - backward_idx):
        return forward_idx
    else:
        return backward_idx

def adjust_audio_segments(note_mappings, audio_file, video_file, output_folder, log_file="adjustment_log.txt", audio_info_log="audio_processing_info.txt"):
    """
    Adjusts and aligns audio and video segments based on note mappings, preserving segment order,
    applies time correction, and retains gaps as original segments without time correction.
    Ensures that video and audio processing steps are consistent and synchronized.
    Uses FFmpeg for faster merging of video and audio streams.

    Args:
        note_mappings (list): List of note mappings containing performance and reference note details.
        audio_file (str): Path to the original audio file.
        video_file (str): Path to the original video file.
        output_folder (str): Folder to save adjusted outputs.
        log_file (str): Path to the log file for debugging.
        audio_info_log (str): Path to the audio processing information log for video alignment.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(log_file, "w") as log, open(audio_info_log, "w") as audio_log:
        log.write("Segment Adjustment Log\n")
        log.write("=" * 50 + "\n")
        audio_log.write("Audio Processing Information Log\n")
        audio_log.write("=" * 50 + "\n")

        try:
            # Load stereo audio
            y, sr = librosa.load(audio_file, sr=None, mono=False)  # Ensure mono=False for stereo
            if y.ndim != 2:
                raise ValueError("Audio is not stereo or was incorrectly loaded.")
            y = y.astype('float32')  # Force conversion to float32
            video_clip = VideoFileClip(video_file)
            log.write(f"Audio loaded: {audio_file}, Sample rate: {sr}, Channels: {y.shape[0]}\n")
            log.write(f"Video loaded: {video_file}\n")
            log.write("=" * 50 + "\n")
        except Exception as e:
            raise RuntimeError(f"Error loading files: {e}")

        # Prepare containers
        segments_and_gaps = []
        last_end_time = 0.0
        crossfade_samples = int(0.01 * sr)  # 0.01 seconds crossfade length
        crossfade_duration = crossfade_samples / sr  # Duration of a single crossfade in seconds
        audio_processing_info = []  # To store details for video alignment

        for mapping in note_mappings:
            try:
                perf_note = mapping.get("performance_note")
                ref_note = mapping.get("reference_note")

                if not perf_note:
                    log.write(f"Skipping mapping with no performance_note.\n")
                    continue

                # Extract timing information
                start_time = perf_note["start"]
                duration = perf_note["duration"]
                end_time = start_time + duration
                time_correction = mapping.get("time_correction", 1.0)

                # Adjust time correction to account for crossfade loss
                adjusted_time_correction = time_correction * (1 + crossfade_duration / duration)

                # Store information in audio processing info log
                audio_processing_info.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "time_correction": adjusted_time_correction
                })
                audio_log.write(f"Start Time: {start_time:.6f}, End Time: {end_time:.6f}, "
                                f"Original Time Correction: {time_correction:.6f}, "
                                f"Adjusted Time Correction: {adjusted_time_correction:.6f}\n")

                start_idx = librosa.time_to_samples(start_time, sr=sr)
                end_idx = librosa.time_to_samples(end_time, sr=sr)

                # Adjust start and end to nearest zero crossing
                start_idx = find_nearest_zero_crossing(y[0], start_idx)
                end_idx = find_nearest_zero_crossing(y[0], end_idx)

                segment_audio = y[:, start_idx:end_idx].astype('float32')

                # Apply time correction for the current segment
                corrected_audio = np.stack([
                    librosa.effects.time_stretch(segment_audio[0], rate=1 / adjusted_time_correction),
                    librosa.effects.time_stretch(segment_audio[1], rate=1 / adjusted_time_correction)
                ]).astype('float32')

                # Handle crossfade if needed
                if len(segments_and_gaps) > 0:
                    previous_audio = segments_and_gaps[-1]["audio"]
                    crossfade_audio = np.zeros((2, crossfade_samples)).astype('float32')
                    for ch in range(2):  # For stereo audio
                        crossfade_audio[ch] = (
                            previous_audio[ch, -crossfade_samples:] * np.linspace(1, 0, crossfade_samples) +
                            corrected_audio[ch, :crossfade_samples] * np.linspace(0, 1, crossfade_samples)
                        )
                    corrected_audio = np.hstack([
                        previous_audio[:, :-crossfade_samples],
                        crossfade_audio,
                        corrected_audio[:, crossfade_samples:]
                    ])

                    # Update previous segment's audio
                    segments_and_gaps[-1]["audio"] = corrected_audio
                    continue  # Skip creating a new segment for this one

                segments_and_gaps.append({
                    "type": "segment",
                    "start_time": start_time,
                    "end_time": end_time,
                    "audio": corrected_audio
                })

            except Exception as e:
                log.write(f"[Error] Processing segment: {e}\n")
                continue

        # Merge all audio segments
        try:
            final_audio = np.hstack([seg["audio"] for seg in segments_and_gaps]).astype('float32')
            output_audio_file = os.path.join(output_folder, "adjusted_output.wav")
            sf.write(output_audio_file, final_audio.T, sr)
            log.write(f"Audio segments merged and saved to: {output_audio_file}\n")
        except Exception as e:
            log.write(f"[Error] Error during audio merging: {e}\n")
            raise RuntimeError(f"Error during audio merging: {e}")

        # Process video based on original time_correction
        try:
            video_segments = []
            for mapping in note_mappings:
                start_time = mapping["performance_note"]["start"]
                duration = mapping["performance_note"]["duration"]
                end_time = start_time + duration
                time_correction = mapping.get("time_correction", 1.0)  # Use original time_correction

                video_segment = video_clip.subclip(start_time, end_time)
                adjusted_video = video_segment.fx(speedx.speedx, factor=(1 / time_correction))  # Use time_correction
                video_segments.append(adjusted_video)

            final_video = concatenate_videoclips(video_segments, method="compose")
            temp_video_file = os.path.join(output_folder, "temp_video.mp4")
            final_video.write_videofile(temp_video_file, codec="libx264", audio=False)
            log.write(f"Video segments merged and saved to temporary file: {temp_video_file}\n")
        except Exception as e:
            log.write(f"[Error] Error during video processing: {e}\n")
            raise RuntimeError(f"Error during video processing: {e}")

        # Combine final video and audio using FFmpeg
        try:
            final_output_file = os.path.join(output_folder, "final_output_with_audio.mp4")
            ffmpeg_path = "/opt/local/bin/ffmpeg"  # Full path to ffmpeg
            ffmpeg_command = [
                ffmpeg_path, "-y", "-i", temp_video_file, "-i", output_audio_file,
                "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", final_output_file
            ]
            subprocess.run(ffmpeg_command, check=True)
            log.write(f"Final adjusted video with audio saved to: {final_output_file}\n")
            log.write("=" * 50 + "\n")
        except Exception as e:
            log.write(f"[Error] Error during final merging: {e}\n")
            raise RuntimeError(f"Error during final merging: {e}")




def show_gui_with_autofit(original_midi, performance_midi):

    original_pm = pm.PrettyMIDI(original_midi)
    performance_pm = pm.PrettyMIDI(performance_midi)

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


    
import tkinter as tk
from tkinter import ttk, simpledialog


def select_file(entry):
    filepath = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filepath)


def select_folder(entry):
    folderpath = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folderpath)



def manual_note_editing(performance_midi_file):
    """
    GUI to manually edit Performance MIDI notes: modify start time, duration, pitch, or delete notes.
    Returns note mappings including relative_offset and time_correction.

    Args:
        performance_midi_file (str): Path to the performance MIDI file.

    Returns:
        list: A list of note mappings with relative_offset and time_correction.
    """
    perf_pm = pm.PrettyMIDI(performance_midi_file)
    original_notes = [
        {"start": note.start, "pitch": note.pitch, "duration": note.end - note.start}
        for inst in perf_pm.instruments for note in inst.notes
    ]

    edited_notes = original_notes.copy()

    note_mappings = []

    root = tk.Tk()
    root.title("Manual Note Editor")

    columns = ("Index", "Start Time", "Pitch", "Duration")
    tree = ttk.Treeview(root, columns=columns, show="headings", height=10)
    for col in columns:
        tree.heading(col, text=col)
    tree.pack(pady=10, padx=10)

    for i, note in enumerate(original_notes):
        tree.insert("", "end", iid=i, values=(i, round(note["start"], 3), note["pitch"], round(note["duration"], 3)))

    ttk.Label(root, text="Start Time:").pack()
    start_entry = ttk.Entry(root)
    start_entry.pack()

    ttk.Label(root, text="Pitch:").pack()
    pitch_entry = ttk.Entry(root)
    pitch_entry.pack()

    ttk.Label(root, text="Duration:").pack()
    duration_entry = ttk.Entry(root)
    duration_entry.pack()

    def on_note_select(event):
        selected = tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        note = edited_notes[idx]
        start_entry.delete(0, tk.END)
        start_entry.insert(0, str(note["start"]))
        pitch_entry.delete(0, tk.END)
        pitch_entry.insert(0, str(note["pitch"]))
        duration_entry.delete(0, tk.END)
        duration_entry.insert(0, str(note["duration"]))

    tree.bind("<<TreeviewSelect>>", on_note_select)

    # 编辑音符
    def edit_note():
        selected = tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a note to edit.")
            return
        idx = int(selected[0])
        try:
            new_start = float(start_entry.get())
            new_pitch = int(pitch_entry.get())
            new_duration = float(duration_entry.get())
            edited_notes[idx] = {"start": new_start, "pitch": new_pitch, "duration": new_duration}
            tree.item(idx, values=(idx, round(new_start, 3), new_pitch, round(new_duration, 3)))
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

    def delete_note():
        selected = tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a note to delete.")
            return
        idx = int(selected[0])
        edited_notes[idx] = None
        tree.delete(idx)

    def confirm_changes():
        nonlocal note_mappings 
        for original, edited in zip(original_notes, edited_notes):
            if edited is None:
                continue
            relative_offset = edited["start"] - original["start"]
            time_correction = edited["duration"] / original["duration"] if original["duration"] > 0 else 1.0
            note_mappings.append({
                "performance_note": original,
                "relative_offset": relative_offset,
                "time_correction": time_correction
            })

        for inst in perf_pm.instruments:
            inst.notes.clear()
            for note in edited_notes:
                if note is not None:
                    inst.notes.append(pm.Note(
                        velocity=100,
                        pitch=note["pitch"],
                        start=note["start"],
                        end=note["start"] + note["duration"]
                    ))
        updated_midi_file = performance_midi_file.replace(".mid", "_manual_edited.mid")
        perf_pm.write(updated_midi_file)
        messagebox.showinfo("Success", f"Updated MIDI saved to {updated_midi_file}")
        root.destroy()

    ttk.Button(root, text="Edit Note", command=edit_note).pack(pady=5)
    ttk.Button(root, text="Delete Note", command=delete_note).pack(pady=5)
    ttk.Button(root, text="Confirm Changes", command=confirm_changes).pack(pady=10)

    root.mainloop()
    return note_mappings

import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def visualize_dtw_3d_interactive(reference_notes, performance_notes, output_html="dtw_visualization_3d.html"):
    """
    Interactive 3D visualization of DTW alignment between reference and performance MIDI notes.

    Args:
        reference_notes (list): List of tuples (start_time, pitch, duration) for reference notes.
        performance_notes (list): List of tuples (start_time, pitch, duration) for performance notes.
        output_html (str): Path to save the interactive HTML visualization.
    """
    # Prepare data
    ref_times = np.array([note[0] for note in reference_notes])
    ref_pitches = np.array([note[1] for note in reference_notes])
    ref_durations = np.array([note[2] for note in reference_notes])
    
    perf_times = np.array([note[0] for note in performance_notes])
    perf_pitches = np.array([note[1] for note in performance_notes])
    perf_durations = np.array([note[2] for note in performance_notes])

    # Create 2D matrices for DTW
    ref_matrix = np.column_stack((ref_pitches, ref_durations))
    perf_matrix = np.column_stack((perf_pitches, perf_durations))
    
    # Compute DTW and alignment path
    distance, path = fastdtw(ref_matrix, perf_matrix, dist=euclidean)
    ref_path, perf_path = zip(*path)

    # Create 3D scatter plot
    fig = go.Figure()

    # Add reference notes
    fig.add_trace(go.Scatter3d(
        x=ref_times, y=ref_pitches, z=ref_durations,
        mode='markers+lines',
        marker=dict(size=5, color='blue'),
        line=dict(color='blue'),
        name='Reference Notes'
    ))

    # Add performance notes
    fig.add_trace(go.Scatter3d(
        x=perf_times, y=perf_pitches, z=perf_durations,
        mode='markers+lines',
        marker=dict(size=5, color='red'),
        line=dict(color='red'),
        name='Performance Notes'
    ))

    # Add DTW alignment lines
    for r_idx, p_idx in path:
        fig.add_trace(go.Scatter3d(
            x=[ref_times[r_idx], perf_times[p_idx]],
            y=[ref_pitches[r_idx], perf_pitches[p_idx]],
            z=[ref_durations[r_idx], perf_durations[p_idx]],
            mode='lines',
            line=dict(color='green', width=2),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Pitch',
            zaxis_title='Duration (s)',
        ),
        title="DTW Alignment Between Reference and Performance MIDI Notes",
        showlegend=True
    )

    # Save interactive plot as an HTML file
    fig.write_html(output_html)
    print(f"Interactive DTW visualization saved to: {output_html}")


import os

def cleanup_files(output_folder, keep_files):

    for root, dirs, files in os.walk(output_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path not in keep_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


def process_and_visualize(video_file, musicxml_file, output_folder, match_mode="automatic"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        cleaned_audio_file = os.path.join(output_folder, "cleaned_audio.wav")
        preprocess_audio(video_file, cleaned_audio_file)

        musicxml_midi_file = os.path.join(output_folder, "musicxml.mid")
        musicxml_to_midi(musicxml_file, musicxml_midi_file)

        frequency, confidence = run_crepe(cleaned_audio_file)

        performance_midi_file = process(
            freqs=frequency,
            conf=confidence,
            audio_path=Path(cleaned_audio_file),
            output_label="performance",
            default_sample_rate=44100
        )
        performance_midi_path = os.path.join(output_folder, os.path.basename(performance_midi_file))
        os.rename(performance_midi_file, performance_midi_path)

        original_pm = pm.PrettyMIDI(musicxml_midi_file)
        performance_pm = pm.PrettyMIDI(performance_midi_path)
        original_notes = [(note.start, note.pitch, note.end - note.start)
                          for inst in original_pm.instruments for note in inst.notes]
        performance_notes = [(note.start, note.pitch, note.end - note.start)
                             for inst in performance_pm.instruments for note in inst.notes]
        
        visualize_dtw_3d_interactive(original_notes, performance_notes, os.path.join(output_folder, "dtw_visualization_3d.html"))

        if match_mode == "manual":
            note_mappings = manual_note_editing(performance_midi_path)
        else:
            bpm = original_pm.estimate_tempo()
            note_mappings = find_matching_notes(performance_notes, original_notes, bpm)

        adjust_audio_segments(note_mappings, cleaned_audio_file, video_file, output_folder)

    except Exception as e:
        print(f"Error: {e}")


def start_gui():
    root = tk.Tk()
    root.title("Belated Metronome")

    frame = ttk.Frame(root)
    frame.pack(pady=10, padx=10)

    audio_label = ttk.Label(frame, text="select your performance audio:")
    audio_label.grid(row=0, column=0, sticky="w")
    audio_entry = ttk.Entry(frame, width=50)
    audio_entry.grid(row=0, column=1, padx=5)
    audio_button = ttk.Button(frame, text="select", command=lambda: select_file(audio_entry))
    audio_button.grid(row=0, column=2)

    musicxml_label = ttk.Label(frame, text="Select your reference musicXML:")
    musicxml_label.grid(row=1, column=0, sticky="w")
    musicxml_entry = ttk.Entry(frame, width=50)
    musicxml_entry.grid(row=1, column=1, padx=5)
    musicxml_button = ttk.Button(frame, text="select", command=lambda: select_file(musicxml_entry))
    musicxml_button.grid(row=1, column=2)

    output_label = ttk.Label(frame, text="Select the output folder:")
    output_label.grid(row=2, column=0, sticky="w")
    output_entry = ttk.Entry(frame, width=50)
    output_entry.grid(row=2, column=1, padx=5)
    output_button = ttk.Button(frame, text="select", command=lambda: select_folder(output_entry))
    output_button.grid(row=2, column=2)

    mode_label = ttk.Label(frame, text="modes:")
    mode_label.grid(row=3, column=0, sticky="w")
    mode_combobox = ttk.Combobox(frame, values=["automatic", "manual"], state="readonly")
    mode_combobox.set("automatic")
    mode_combobox.grid(row=3, column=1, padx=5)

    process_button = ttk.Button(
        frame,
        text="process",
        command=lambda: process_and_visualize(
            audio_entry.get(), musicxml_entry.get(), output_entry.get(), mode_combobox.get()
        )
    )
    process_button.grid(row=4, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    start_gui()
