import io
import time
from fastapi import FastAPI, Response
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# sft usage
print(cosyvoice.list_avaliable_spks())
app = FastAPI()


# streaming response
def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))
# streaming response


### clone_eq & clone_eq_other_lan & tts

### helper

def base64_to_wav(encoded_str, output_path):
    if not encoded_str:
        raise ValueError("Base64 encoded string is empty.")

    # 将base64编码的字符串解码为字节
    wav_bytes = base64.b64decode(encoded_str)

    # 检查输出路径是否存在，如果不存在则创建
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 将解码后的字节写入文件
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_bytes)

    print(f"WAV file has been saved to {output_path}")


# 获取请求参数
def get_params(req):
    params={
        "text":"",
        "lang":"",
        "role":"中文女",
        "reference_audio":None,
        "reference_text":""
    }
    # 原始字符串
    params['text'] = req.args.get("text","").strip() or req.form.get("text","").strip()
    
    # 字符串语言代码
    params['lang'] = req.args.get("lang","").strip().lower() or req.form.get("lang","").strip().lower()
    # 兼容 ja语言代码
    if params['lang']=='ja':
        params['lang']='jp'
    elif params['lang'][:2] == 'zh':
        # 兼容 zh-cn zh-tw zh-hk
        params['lang']='zh'
    
    # 角色名 
    role = req.args.get("role","").strip() or req.form.get("role",'')
    if role:
        params['role']=role
    
    # 要克隆的音色文件    
    params['reference_audio'] = req.args.get("reference_audio",None) or req.form.get("reference_audio",None)
    encode=req.args.get('encode','') or req.form.get('encode','')
    if  encode=='base64':
        tmp_name=f'tmp/{time.time()}-clone-{len(params["reference_audio"])}.wav'
        base64_to_wav(params['reference_audio'],root_dir+'/'+tmp_name)
        params['reference_audio']=tmp_name
    # 音色文件对应文本
    params['reference_text'] = req.args.get("reference_text",'').strip() or req.form.get("reference_text",'')
    
    return params


def del_tmp_files(tmp_files: list):
    print('正在删除缓存文件...')
    for f in tmp_files:
        if os.path.exists(f):
            print('删除缓存文件:', f)
            os.remove(f)


# 实际批量合成完毕后连接为一个文件
def batch(tts_type,outname,params):
    text=params['text'].strip().split("\n")
    text=[t.replace("。",'，') for t in text]
    if len(text)>1 and not shutil.which("ffmpeg"):
        raise Exception('多行文本合成必须安装 ffmpeg')
    
    # 按行合成
    out_list=[]
    prompt_speech_16k=None
    if tts_type!='tts':
        if not params['reference_audio'] or not os.path.exists(f"{root_dir}/{params['reference_audio']}"):
            raise Exception(f'参考音频未传入或不存在 {params["reference_audio"]}')
        prompt_speech_16k = load_wav(params['reference_audio'], 16000)
    for i,t in enumerate(text):
        if not t.strip():
            continue
        tmp_name=f"{tmp_dir}/{time.time()}-{i}-{tts_type}.wav"
        print(f'{t=}\n{tmp_name=},\n{tts_type=}\n{params=}')
        if tts_type=='tts':
            # 仅文字合成语音
            output = tts_model.inference_sft(t, params['role'],stream=False)
        elif tts_type=='clone_eq':
            # 同语言克隆
            output=clone_model.inference_zero_shot(t,params['reference_text'], prompt_speech_16k)
        else:
            output = clone_model.inference_cross_lingual(f'<|{params["lang"]}|>{t}', prompt_speech_16k)
        try:
            torchaudio.save(tmp_name, output['tts_speech'], 22050)
        except TypeError as e:
            torchaudio.save(tmp_name, list(output)[0]['tts_speech'], 22050)
        out_list.append(tmp_name)
    if len(out_list)==0:
        raise Exception('合成失败')
    if len(out_list)==1:
        print(f"音频文件生成成功：{out_list[0]}")
        return out_list[0]
    # 将 多个音频片段连接
    txt_tmp="\n".join([f"file '{it}'" for it in out_list])
    txt_name=f'{time.time()}.txt'
    with open(f'{tmp_dir}/{txt_name}','w',encoding='utf-8') as f:
        f.write(txt_tmp)
    out_list.append(f'{tmp_dir}/{txt_name}')
    try:
        subprocess.run(["ffmpeg","-hide_banner", "-ignore_unknown","-y","-f","concat","-safe","0","-i",f'{tmp_dir}/{txt_name}',"-c:a","copy",tmp_dir + '/' + outname],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   encoding="utf-8",
                   check=True,
                   text=True,
                   creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        del_tmp_files(out_list)
        print(e)
        raise
    else:
        del_tmp_files(out_list)
        print(f"音频文件生成成功：{tmp_dir}/{outname}")
        return tmp_dir + '/' + outname
### helper

# 定义请求参数模型
class TTSParams:
    text: str
    lang: str = None
    role: str = "中文女"
    reference_audio: str = None
    reference_text: str = None

# 定义路由和视图函数
@app.post("/tts")
async def tts(params: TTSParams):
    if not params.text:
        raise HTTPException(status_code=500, detail="缺少待合成的文本")
    try:
        outname = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}-tts.wav"
        outname = batch('tts', outname, params.__dict__)
        return FileResponse(outname, media_type='audio/x-wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone_eq")
async def clone_eq(params: TTSParams):
    global clone_model
    if not clone_model:
        clone_model = CosyVoice('pretrained_models/CosyVoice-300M')
    if not params.text:
        raise HTTPException(status_code=500, detail="缺少待合成的文本")
    if not params.reference_text:
        raise HTTPException(status_code=500, detail="必须设置参考音频对应的参考文本reference_text")
    try:
        outname = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}-clone_eq.wav"
        outname = batch('clone_eq', outname, params.__dict__)
        return FileResponse(outname, media_type='audio/x-wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone_mul")
async def clone_mul(params: TTSParams):
    global clone_model
    if not clone_model:
        clone_model = CosyVoice('pretrained_models/CosyVoice-300M')
    if not params.text:
        raise HTTPException(status_code=500, detail="缺少待合成的文本")
    if not params.lang:
        raise HTTPException(status_code=500, detail="必须设置待合成文本的语言代码")
    try:
        outname = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}-clone_mul.wav"
        outname = batch('clone_mul', outname, params.__dict__)
        return FileResponse(outname, media_type='audio/x-wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/tts")
async def tts(query: str):
    start = time.process_time()
    output = cosyvoice.inference_sft(query, '中文女')
    end = time.process_time()
    print(f"Infer time: {end-start:.1f}s")
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")

if __name__ == '__main__':
    uvicorn.run(app,
                host=None,
                port=8000,
                log_level="debug")