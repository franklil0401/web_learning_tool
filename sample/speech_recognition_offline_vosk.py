import os
import queue
import json

import sounddevice as sd
import vosk


# 本地中文模型路径（请根据你实际解压位置修改）
MODEL_PATH = r"e:\business\vosk-model-cn-0.22"


def main():
    if not os.path.isdir(MODEL_PATH):
        print("未找�? Vosk 中文模型目录�?", MODEL_PATH)
        print("请先在浏览器搜索并下�? `vosk-model-small-cn-0.22`�?")
        print("解压后把文件夹路径改成上面的 MODEL_PATH，或在这里改成你的实际路径�?")
        return

    print("正在加载离线中文识别模型，请稍�?...")
    model = vosk.Model(MODEL_PATH)

    samplerate = 16_000  # Vosk 推荐采样�?
    audio_queue: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time, status):
        # 回调里只做“把数据放进队列”这一件事，避免阻�?
        if status:
            print("输入状态：", status, flush=True)
        audio_queue.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            recognizer = vosk.KaldiRecognizer(model, samplerate)
            print("离线语音识别开始，�? Ctrl+C 结束�?")
            print("你可以一直说话，")
            print(" - 说话过程中，会不断输出【临时识别】结果；")
            print(" - 你短暂停顿时，会输出这一小段的【最终识别】结果�?")

            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    # 一段完整语音结束，给出最终结�?
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print("【最终识别�?", text)
                else:
                    # 连续语音时，给出中间的临时识别结�?
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        print("【临时识别�?", partial)

    except KeyboardInterrupt:
        print("\n检测结束�?")
    except Exception as e:
        print("发生错误�?", repr(e))


if __name__ == "__main__":
    main()