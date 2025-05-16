from fastapi import FastAPI, BackgroundTasks, Form
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import os
from lite_avatar import liteAvatar
from typing import Optional
from fastapi.responses import JSONResponse

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)
task_store = {}  # {task_id: {"status": ..., "result": ...}}

# 任务执行函数
def run_avatar_task(task_id, data_dir, audio_file_path, result_dir):
    try:
        task_store[task_id]["status"] = "running"
        avatar = liteAvatar(data_dir=data_dir, num_threads=1, generate_offline=True)
        
        # 添加音频文件处理逻辑
        processed_audio_path = audio_file_path
        
        # 检查WAV文件格式
        try:
            import wave
            with wave.open(audio_file_path, 'rb') as wav_file:
                # 如果能成功打开，说明是标准PCM格式，无需转换
                pass
        except wave.Error as e:
            # 检测到非标准格式，尝试用scipy转换
            try:
                # 创建一个临时文件路径用于保存转换后的音频
                import tempfile
                import os
                temp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{os.path.basename(audio_file_path)}")
                
                # 使用scipy读取并转换音频
                from scipy.io import wavfile
                import numpy as np
                
                # 读取音频数据
                sample_rate, data = wavfile.read(audio_file_path)
                
                # 如果是浮点格式，转换为16位整数PCM
                if data.dtype.kind == 'f':
                    # 缩放到[-32768, 32767]范围
                    data = np.int16(np.clip(data * 32767, -32768, 32767))
                
                # 写入临时文件
                wavfile.write(temp_audio_path, sample_rate, data)
                processed_audio_path = temp_audio_path
                print(f"Converted audio file from {audio_file_path} to {processed_audio_path}")
                
            except Exception as conversion_error:
                # 如果scipy转换失败，尝试使用ffmpeg
                try:
                    import subprocess
                    import tempfile
                    import os
                    
                    temp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{os.path.basename(audio_file_path)}")
                    
                    # 使用ffmpeg转换
                    cmd = [
                        'ffmpeg', '-y', '-i', audio_file_path, 
                        '-acodec', 'pcm_s16le', '-ar', '24000', # 使用与原始文件相同的采样率
                        temp_audio_path
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    processed_audio_path = temp_audio_path
                    print(f"Converted audio file using ffmpeg from {audio_file_path} to {processed_audio_path}")
                    
                except Exception as ffmpeg_error:
                    print(f"All conversion methods failed. Original error: {e}, Scipy error: {conversion_error}, FFmpeg error: {ffmpeg_error}")
                    raise Exception(f"Unable to convert audio format: {e}")
        
        # 使用处理后的音频文件
        avatar.handle(processed_audio_path, result_dir)
        
        # 结果路径
        result_video = os.path.join(result_dir, "test_demo.mp4")
        if os.path.exists(result_video):
            task_store[task_id]["status"] = "completed"
            task_store[task_id]["result"] = result_video
        else:
            task_store[task_id]["status"] = "failed"
            task_store[task_id]["result"] = "result video not found"
    except Exception as e:
        import traceback
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["result"] = error_message

class AvatarTaskRequest(BaseModel):
    data_dir: str
    result_dir: str
    audio_file: str

@app.post("/submit-task")
def submit_task(
    data_dir: str = Form(...),
    result_dir: str = Form(...),
    audio_file: str = Form(...)
):
    task_id = str(uuid.uuid4())
    task_store[task_id] = {"status": "pending", "result": None}
    # 校验音频文件是否存在
    if not os.path.isfile(audio_file):
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["result"] = f"audio_file not found: {audio_file}"
        return {"task_id": task_id, "error": f"audio_file not found: {audio_file}"}
    # 提交任务
    executor.submit(run_avatar_task, task_id, data_dir, audio_file, result_dir)
    return {"task_id": task_id}

@app.get("/task-status/{task_id}")
def task_status(task_id: str):
    task = task_store.get(task_id)
    if not task:  
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return {"status": task["status"], "result": task["result"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50001)
