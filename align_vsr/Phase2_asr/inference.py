import sys
sys.path.append('/work/liuzehua/task/VSR/cnvsrc')
import os
import hydra
from vsr2asr.model5.Phase2_asr.transforms import TextTransform
import json
import multiprocessing
import logging
def filelist(listcsv, text_transform, cfg):
    fns = []
    lines = []
    uid2content = {}
    with open(listcsv) as fp:
        lines = fp.readlines()
    root = cfg.audio_data_root_dir + '/' + lines[0].split(',')[0]
    for line in lines:
        _, vfn, _, tokens = line.strip().split(',')
        uid = os.path.join(vfn)
        fn = f"{root}/{vfn.replace('//', '/')}"
        uid2content[uid] = text_transform.post_process([int(t) for t in tokens.split(' ')])
        if os.path.exists(fn):
            fns.append((fn,uid))
        print(fn)
    return fns, uid2content
        
def task_on_gpu(gpu_id, cfg, fns, target_content, return_dic, task_id):
            import torch
            import os
            # 设置CUDA设备
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = torch.device("cuda")
            print(f"Running on {torch.cuda.get_device_name(device)}")
            import soundfile as sf
            import os
            import torchvision, torchmetrics
            from vsr2asr.model5.Phase2_asr.asr_model import ASR
            from vsr2asr.model5.Phase2_asr.transforms import TextTransform
            from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
            from espnet.nets.batch_beam_search import BatchBeamSearch
            from espnet.nets.lm_interface import dynamic_import_lm
            from espnet.nets.scorers.length_bonus import LengthBonus
            import torchaudio
            from transformers import (
                Wav2Vec2FeatureExtractor,
            )
            def compute_word_level_distance(ref, hyp):
                # 将字符串转换为小写并分割成单词列表
                ref_words = ref.lower().split()
                hyp_words = hyp.lower().split()

                # 计算编辑距离（即错误的个数）
                errors = torchaudio.functional.edit_distance(ref_words, hyp_words)

                # 总的单词数
                total_words = len(ref_words)

                # 计算WER
                wer = errors / total_words if total_words > 0 else float('inf')

                return wer, errors, total_words



            def get_beam_search_decoder(
                model,
                token_list,
                rnnlm=None,
                rnnlm_conf=None,
                penalty=0,
                ctc_weight=0.1,
                lm_weight=0.0,
                beam_size=40,
            ):
                sos = model.odim - 1
                eos = model.odim - 1
                scorers = model.scorers()

                if not rnnlm:
                    lm = None
                else:
                    lm_args = get_model_conf(rnnlm, rnnlm_conf)
                    lm_model_module = getattr(lm_args, "model_module", "default")
                    lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
                    lm = lm_class(len(token_list), lm_args)
                    torch_load(rnnlm, lm)
                    lm.eval()

                scorers["lm"] = lm
                scorers["length_bonus"] = LengthBonus(len(token_list))
                weights = {
                    "decoder": 1.0 - ctc_weight,
                    "ctc": ctc_weight,
                    "lm": lm_weight,
                    "length_bonus": penalty,
                }

                return BatchBeamSearch(
                    beam_size=beam_size,
                    vocab_size=len(token_list),
                    weights=weights,
                    scorers=scorers,
                    sos=sos,
                    eos=eos,
                    token_list=token_list,
                    pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
                )

            
            device = cfg.device
            text_transform = TextTransform()
            token_list = text_transform.token_list # word to id

            # load model

            model = ASR(len(token_list), cfg).to(device)#len token_list , so input unigram file is very ok
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.hubert_model)

            # load checkpoint
            if cfg.infer_ckpt_path.endswith('pth'):#.pth can be directly saved 
                logging.info('进行预测的模型： ' + str(cfg.infer_ckpt_path))
                model.load_state_dict(
                    torch.load(cfg.infer_ckpt_path, map_location=device),
                    strict = False
                )
            else:
                ckpt = torch.load(cfg.infer_ckpt_path, map_location=device)#.ckpt need extract the state_dict to load model
                logging.info('进行预测的模型： ' + str(cfg.infer_ckpt_path))
                state_dict = {}
                for key, value in ckpt['state_dict'].items():
                    if key[:6] == 'model.':
                        state_dict[key[6:]] = value
                model.load_state_dict(state_dict, strict=False)

            # predict
            model.eval()
            infos = {}
            beam_search = get_beam_search_decoder(model, token_list, ctc_weight=0.1,)
            total = 0
            error = 0
            for i, (fn, uid) in enumerate(fns):
                wav, sr = sf.read(fn)
                audio = feature_extractor(wav, return_tensors="pt", sampling_rate = sr).input_values
                audio = audio.to(device)
                with torch.no_grad():
                    B,T = audio.size()

                    x = model.hubert_encoder(audio).last_hidden_state

                    # Reshape x to (batch_size * time_steps, embedding_size)
                    batch_size, time_steps, embedding_size = x.size()
                    x_reshaped = x.view(batch_size * time_steps, embedding_size).detach().cpu()

                    # Predict cluster center indices for each sample using the k-means model
                    labels = model.kmeans.predict(x_reshaped)

                    # Convert labels to a PyTorch tensor and ensure it is on the same device as the original tensor
                    labels_tensor = torch.tensor(labels, dtype=torch.long, device=x.device).view(batch_size, time_steps)

                    audio_feature = model.compact_audio_memory(labels_tensor)

                    #audio mask
                    B,T,_ = audio_feature.size()    
                    audio_feature, _ = model.asr_frontend(audio_feature, None)
                    audio_feature = audio_feature.squeeze(0)

                    nbest_hyps = beam_search(audio_feature)
                    nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]#select the best
                    predicted = add_results_to_json(nbest_hyps, token_list)
                    transcript = predicted.replace("▁", " ").strip().replace("<eos>", "")
                
                    target = target_content[uid]
                    cur_wer, cur_error, cur_total = compute_word_level_distance(target, transcript)
                    total = total + cur_total
                    error = error + cur_error
                    transcript = transcript.lower()
                    target = target.lower()
                    print(f"task_id {task_id} {uid}-pred: {transcript}")
                    print(f"task_id {task_id} {uid}-targ: {target}")
                    print(f'task_id {task_id} {i+1}/{len(fns)} wer:{str(cur_wer*100)}%')
                    print(f'task_id {task_id} {i+1}/{len(fns)} avg_wer:{str(error*100/total)}%')
                    infos[uid] = {
                        'fn' : fn,
                        'pred': transcript,
                        'targ': target,
                        'wer' : str(cur_wer) + '%'
                    }
                
            logging.info(f'task_id {task_id} total-wer:{error} / {total} = {cur_error/cur_total*100}%')

            return_dic[task_id] = (error, total, infos)




@hydra.main(config_path="/work/liuzehua/task/VSR/cnvsrc/conf/vsr2asr/model5/Phase2_asr", config_name="inference_asr")
def main(cfg):
    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    os.makedirs(cfg.save.save_path, exist_ok = True)
    # init text
    text_transform = TextTransform()
    valid_csv = os.path.join(cfg.code_root_dir, cfg.data.dataset.label_dir, cfg.data.dataset.val_file)
    fns, target_content = filelist(valid_csv, text_transform, cfg)
    logging.info('预测的valid_csv:' + str(valid_csv) )



    # 您所拥有的GPU列表
    available_gpus = [0,1,2,3,4,5]
    gpu_process_num = [3,0,0,0,3,3]
    split_num = sum(gpu_process_num)

            
    # 使用Python的多进程
    processes = []
    last = 0
    count = 0

    task_path_list = fns 
    for i in range(len(available_gpus)):
        for j in range(gpu_process_num[i]):

            avg = len(task_path_list) // split_num
            remain = len(task_path_list) % split_num
            size = avg + 1 if count < remain else avg
            temp_input = task_path_list[last:last + size]
            last += size
            gpu_id = available_gpus[i]
            count = count + 1
            p = multiprocessing.Process(target=task_on_gpu, args=(gpu_id, cfg, temp_input, target_content, return_dict, count))
            p.start()
            processes.append(p)
            
    # 等待所有进程完成
    for p in processes:
        p.join()


    # 收集所有结果
    sum_infos = {}
    sum_error = 0
    sum_total = 0
    for key in return_dict:
        error, total, infos = return_dict[key]
        sum_infos.update(infos)
        sum_error = sum_error + error
        sum_total = sum_total + total
    dic_wer = {'total_wer': str(float(100 * sum_error / sum_total)) + '%' }

    logging.info(f'total wer: {sum_error*100 / sum_total}%')

    with open(cfg.save.save_json, 'w', encoding='utf-8') as f:
        json.dump({**dic_wer, **sum_infos}, f, ensure_ascii=False, indent=4)

   


  
if __name__ == '__main__': 
    main()


