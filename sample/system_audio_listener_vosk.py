# -*- coding: utf-8 -*-
import os
import queue
import json
import time
from array import array

import sounddevice as sd
import vosk

# Local Chinese model path (same as online voice package)
MODEL_PATH = r"e:/business/vosk-model-cn-0.22"

# Live RAG: final recognition ?? chunk + write to FAISS (need pip install sentence-transformers)
# Disable: environment variable ENABLE_LIVE_RAG=0
_rag_import_warned = False


def _live_rag_index_teacher(text: str) -> None:
    """Each final recognition text: JSONL + vector chunk incremental indexing."""
    global _rag_import_warned
    if not (text or "").strip():
        return
    flag = os.environ.get("ENABLE_LIVE_RAG", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return
    try:
        from rag.hooks import on_teacher_text

        on_teacher_text(text.strip(), index_immediately=True)
    except Exception as e:
        if not _rag_import_warned:
            print(
                "[RAG] Vector not imported (please install: pip install sentence-transformers):",
                repr(e),
                flush=True,
            )
            _rag_import_warned = True


def _load_model() -> vosk.Model:
    if not os.path.isdir(MODEL_PATH):
        print("Vosk Chinese model directory not found:", MODEL_PATH)
        print("Please make sure the model is downloaded and set MODEL_PATH to the actual directory.")
        raise SystemExit(1)

    print("Loading offline Chinese recognition model, please wait...")
    return vosk.Model(MODEL_PATH)


def _find_capture_device() -> int:
    """
    Priority use Windows current 'Default Recording Device', which is convenient for external microphones and system settings.
    If default fails, try devices with names containing 'Stereo Mix' or similar.
    """
    devices = sd.query_devices()
    default_in = sd.default.device[0]

    if default_in is not None and int(default_in) >= 0:
        dev = devices[int(default_in)]
        if dev.get("max_input_channels", 0) > 0:
            print(f"Using default recording device: #{default_in} - {dev['name']}")
            return int(default_in)

    keywords = ["stereo mix"]
    for idx, dev in enumerate(devices):
        name_lower = str(dev["name"]).lower()
        if dev["max_input_channels"] > 0 and any(k in name_lower for k in keywords):
            print(f"Using system recording device (Stereo Mix): #{idx} - {dev['name']}")
            return idx

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"Default/Stereo Mix not found, using input device: #{idx} - {dev['name']}")
            return idx

    print("No available input device found, cannot record.")
    raise SystemExit(1)


def main():
    model = _load_model()

    audio_queue: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print("Input status:", status, flush=True)
        audio_queue.put(bytes(indata))

    device_index = _find_capture_device()
    dev_info = sd.query_devices(device_index)
    in_channels = dev_info.get("max_input_channels", 1) or 1

    samplerate = int(dev_info.get("default_samplerate", 16_000) or 16_000)
    print(f"Using sample rate: {samplerate} Hz, channels: {in_channels}")

    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=8000,
            dtype="int16",
            channels=in_channels,
            device=device_index,
            callback=callback,
        ):
            recognizer = vosk.KaldiRecognizer(model, samplerate)
            print("System audio offline recognition started, press Ctrl+C to stop.")
            print("Note: When default recording device is 'Stereo Mix', system playback sound can be recorded; otherwise, microphone sound is recorded.")

            partial_refresh_sec = 0.4
            last_partial_print_at = 0.0
            last_partial_text = ""
            printed_partial_line = False

            while True:
                data = audio_queue.get()

                if in_channels > 1:
                    samples = array("h")
                    samples.frombytes(data)
                    if len(samples) >= in_channels:
                        mono = array("h")
                        step = in_channels
                        for i in range(0, len(samples) - (len(samples) % step), step):
                            s = 0
                            for c in range(step):
                                s += samples[i + c]
                            mono.append(int(s / step))
                        data = mono.tobytes()

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if printed_partial_line:
                        print()
                        printed_partial_line = False
                        last_partial_text = ""

                    if text:
                        print("[System Audio - Final Recognition]", text)
                        _live_rag_index_teacher(text)
                else:
                    now = time.time()
                    if now - last_partial_print_at >= partial_refresh_sec:
                        partial = (
                            json.loads(recognizer.PartialResult())
                            .get("partial", "")
                            .strip()
                        )
                        if partial and partial != last_partial_text:
                            print(
                                "\r[System Audio - Real-time] " + partial + " " * 10,
                                end="",
                                flush=True,
                            )
                            printed_partial_line = True
                            last_partial_text = partial
                            last_partial_print_at = now

    except KeyboardInterrupt:
        print("\nSystem audio monitoring ended.")
    except Exception as e:
        print("System audio module error:", repr(e))


if __name__ == "__main__":
    main()
