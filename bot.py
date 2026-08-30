import os, time, cv2, torch, asyncio, tempfile, subprocess, threading
import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CREDIT = "This bot made by t.me/remotecbr"
TARGET_SIZE = 512
MAX_SECONDS = 10

# GitHub Actions: 5.5 ghante baad auto-shutdown (agle run me khud restart ho jayega)
def _auto_shutdown():
    time.sleep(5.5 * 60 * 60)
    print("Auto-shutdown (restart cycle)")
    os._exit(0)
threading.Thread(target=_auto_shutdown, daemon=True).start()

print("Loading model...")
print(CREDIT)
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",
                       pretrained="face_paint_512_v2", trust_repo=True).eval()
torch.set_num_threads(2)
print("Model ready!")

def fit_dims(w, h):
    scale = TARGET_SIZE / max(w, h)
    return max(int(w*scale)//8*8, 8), max(int(h*scale)//8*8, 8)

@torch.inference_mode()
def anime_frame(frame_bgr, size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize(size)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2,0,1).unsqueeze(0)/127.5 - 1
    y = model(x)
    out = ((y[0].permute(1,2,0).numpy()+1)/2*255).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def convert_video(in_path, silent_path, out_path, status):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_WIDTH)), int(cap.get(cv2.CAP_PROP_HEIGHT))
    size = fit_dims(w, h)
    max_frames = int(fps * MAX_SECONDS)

    writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    done = 0
    while done < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(anime_frame(frame, size))
        done += 1
        if done % 30 == 0:
            status(f"🎨 {done} frames processed...")
    cap.release()
    writer.release()

    subprocess.run(["ffmpeg","-y","-i",silent_path,"-i",in_path,
                    "-map","0:v:0","-map","1:a:0?",
                    "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
                    "-shortest", out_path], check=True, capture_output=True)
    return done

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome!\n\n"
        "Send me any video (max 20MB / first 10 seconds) and I will convert it to anime style.\n\n"
        f"✨ {CREDIT}"
    )

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📥 Downloading video...")
    loop = asyncio.get_running_loop()
    file = await update.message.video.get_file()

    with tempfile.TemporaryDirectory() as tmp:
        inp, silent, out = (os.path.join(tmp, f) for f in ("in.mp4", "silent.mp4", "out.mp4"))
        await file.download_to_drive(inp)
        await msg.edit_text("🎨 Converting to anime style... (this may take a while on CPU)")

        def status(text):
            asyncio.run_coroutine_threadsafe(msg.edit_text(text), loop)

        frames = await asyncio.to_thread(convert_video, inp, silent, out, status)
        await msg.edit_text(f"✅ Done! {frames} frames converted. Sending video...")
        with open(out, "rb") as f:
            await update.message.reply_video(
                f,
                supports_streaming=True,
                caption=f"🎨 Anime Style Conversion\n\n{CREDIT}"
            )

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    print("Bot is running...")
    print(CREDIT)
    app.run_polling()

if __name__ == "__main__":
    main()
