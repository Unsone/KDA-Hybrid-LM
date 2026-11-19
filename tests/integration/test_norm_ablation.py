import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_fixed_lr_ablation_reuses_baselines_and_records_factorial_contrasts(tmp_path):
    data=tmp_path/'data'
    data.mkdir()
    tokens=np.tile(np.arange(8,dtype=np.uint16),20)
    metadata=dict(vocab_size=8,token_counts={},sha256={},tokenizer_sha256='fixture')
    for split in ('train','validation','test'):
        np.save(data/f'{split}.npy',tokens)
        metadata['token_counts'][split]=len(tokens)
        metadata['sha256'][split]=hashlib.sha256(tokens).hexdigest()
    (data/'metadata.json').write_text(json.dumps(metadata))
    reference,output=tmp_path/'reference',tmp_path/'ablation'
    subprocess.run([
        sys.executable,str(ROOT/'scripts/train_lm.py'),'--mode','compare',
        '--optimizers','adam','adam_s','--data-dir',str(data),'--output',str(reference),
        '--device','cpu','--steps','2','--search-steps','1','--seeds','42','43',
        '--adam-lrs','0.001','--scaled-lrs','0.03','--batch-size','2',
        '--accumulation','1','--length','4','--layers','1','--hidden-dim','8',
        '--intermediate-dim','16','--heads','2','--eval-every','1','--eval-batches','1','--no-plots',
    ],check=True,capture_output=True,text=True)
    command=[sys.executable,str(ROOT/'scripts/run_norm_ablation.py'),
             '--reference',str(reference),'--data-dir',str(data),'--output',str(output),
             '--device','cpu','--no-plots']
    subprocess.run(command,check=True,capture_output=True,text=True)
    trained=json.loads((output/'summary.json').read_text())
    combined=json.loads((output/'comparison_summary.json').read_text())
    assert len(trained)==4 and {r['arm'] for r in trained}=={'B','C'}
    assert len(combined)==8 and sum(r['reused'] for r in combined)==4
    assert not (output/'search.json').exists()
    for seed in [42,43]:
        assert len({r['initial_sha256'] for r in combined if r['seed']==seed})==1
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['search_budget']==0
    assert protocol['arms']['B']['norm_lr']==0.001
    assert protocol['arms']['C']['norm_lr']==0.03
    rows=list(csv.DictReader((output/'test_contrasts.csv').open(encoding='utf-8-sig')))
    by_arm={(r['arm'],r['seed']):r for r in combined}
    for row in rows:
        if row['contrast']=='D-B-C+A':
            seed=int(row['seed'])
            value=sum(w*by_arm[a,seed]['test']['nll'] for a,w in [('D',1),('B',-1),('C',-1),('A',1)])
            assert abs(float(row['test_nll_delta'])-value)<1e-12
    # 正式入口也可以从头训练四组；A/D 与复用结果应完全一致。
    rerun=tmp_path/'rerun'
    command[command.index(str(output))]=str(rerun)
    subprocess.run(command+['--arms','A','B','C','D'],check=True,capture_output=True,text=True)
    fresh=json.loads((rerun/'summary.json').read_text())
    assert len(fresh)==8 and not any(r['reused'] for r in fresh)
    for row in fresh:
        assert row['test']==by_arm[row['arm'],row['seed']]['test']
    # 旧实验实际 LR 与方案不符时，不能将其当成 A/D 复用。
    old=json.loads((reference/'summary.json').read_text())
    old[0]['lr']=0.002
    (reference/'summary.json').write_text(json.dumps(old))
    rejected=subprocess.run(command+['--arms','A','B','C','D'],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'planned fixed LR' in rejected.stderr
