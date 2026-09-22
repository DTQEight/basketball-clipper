import json
from pathlib import Path

files = {
    ('qwen', '原版'): Path(r'E:\basketball-project\training\vlm_eval_results.jsonl'),
    ('qwen', '加密'): Path(r'E:\basketball-project\training\vlm_eval_qwen_dense.jsonl'),
    ('ds', '原版'): Path(r'E:\basketball-project\training\vlm_eval_deepseek.jsonl'),
    ('ds', '加密'): Path(r'E:\basketball-project\training\vlm_eval_deepseek_dense.jsonl'),
}

def load(p):
    rows = {}
    for l in p.read_text(encoding='utf-8').splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r['event_id']] = r
    return rows

data = {k: load(p) for k, p in files.items()}
common_ids = sorted(set(data[('qwen', '原版')]) & set(data[('qwen', '加密')])
                    & set(data[('ds', '原版')]) & set(data[('ds', '加密')]))
print(f'严格共同事件: {len(common_ids)} 个（同样本同 baseline）')


def metric(rows, ids):
    sub = [rows[i] for i in ids if i in rows and rows[i].get('vlm_verdict') in ('goal', 'miss')]
    g = [r for r in sub if r.get('band') == 'gray']
    g_hit = sum(1 for r in g if (r['vlm_verdict'] == 'goal') == (r['label'] == 1))
    gp = [r for r in g if r['label'] == 1]
    gp_hit = sum(1 for r in gp if r['vlm_verdict'] == 'goal')
    gn = [r for r in g if r['label'] == 0]
    gn_hit = sum(1 for r in gn if r['vlm_verdict'] == 'miss')
    ex = [r for r in g if 0.2 <= r['oof'] <= 0.8]
    ex_hit = sum(1 for r in ex if (r['vlm_verdict'] == 'goal') == (r['label'] == 1))
    dmx = [r for r in sub if r['vlm_verdict'] == 'miss' and r['oof'] < 0.5]
    dmx_ok = sum(1 for r in dmx if r['label'] == 0)
    dmg = [r for r in sub if r['vlm_verdict'] == 'goal' and r['oof'] >= 0.5]
    dmg_ok = sum(1 for r in dmg if r['label'] == 1)
    h = [r for r in sub if r['band'] == 'auto_keep']
    lx = [r for r in sub if r['band'] == 'auto_reject']
    h_hit = sum(1 for r in h if (r['vlm_verdict'] == 'goal') == (r['label'] == 1))
    l_hit = sum(1 for r in lx if (r['vlm_verdict'] == 'miss') == (r['label'] == 0))
    return dict(
        n=len(sub), gray_acc=(g_hit / len(g) * 100 if g else 0),
        gray_n=len(g), gp_hit=gp_hit, gp_n=len(gp),
        gn_hit=gn_hit, gn_n=len(gn),
        ex_acc=(ex_hit / len(ex) * 100 if ex else 0), ex_n=len(ex),
        dmx_pre=(dmx_ok / len(dmx) * 100 if dmx else None), dmx_n=len(dmx),
        dmg_pre=(dmg_ok / len(dmg) * 100 if dmg else None), dmg_n=len(dmg),
        two_ends_acc=((h_hit + l_hit) / max(len(h) + len(lx), 1) * 100
                      if (h or lx) else None),
        two_ends_n=len(h) + len(lx),
    )


def fmt_pct(v):
    if v is None:
        return "?"
    return f"{v:.0f}%"


def row(name, m):
    print(f'{name:24s} | 灰区 {m["gray_acc"]:.1f}%/n={m["gray_n"]:3d} | '
          f'真进球召回 {m["gp_hit"]}/{m["gp_n"]}='
          f'{m["gp_hit"] / max(m["gp_n"], 1) * 100:.0f}% | '
          f'误报识别 {m["gn_hit"]}/{m["gn_n"]}='
          f'{m["gn_hit"] / max(m["gn_n"], 1) * 100:.0f}% | '
          f'极灰 {m["ex_acc"]:.0f}%/n={m["ex_n"]:2d} | '
          f'双判× {fmt_pct(m["dmx_pre"])}/{m["dmx_n"]:2d} | '
          f'双判√ {fmt_pct(m["dmg_pre"])}/{m["dmg_n"]:2d}')


for mname in ('qwen', 'ds'):
    label = 'Qwen3-VL-Plus' if mname == 'qwen' else 'DeepSeek-V4-Flash-V'
    print()
    print('─' * 140)
    print(label)
    print('─' * 140)
    m1 = metric(data[(mname, '原版')], common_ids)
    m2 = metric(data[(mname, '加密')], common_ids)
    row('原版 16 帧', m1)
    row('加密 21 帧', m2)
    delta_acc = m2['gray_acc'] - m1['gray_acc']
    delta_ex = m2['ex_acc'] - m1['ex_acc']
    delta_recall = (m2['gp_hit'] / max(m2['gp_n'], 1)
                    - m1['gp_hit'] / max(m1['gp_n'], 1)) * 100
    delta_mis = (m2['gn_hit'] / max(m2['gn_n'], 1)
                 - m1['gn_hit'] / max(m1['gn_n'], 1)) * 100
    sign = lambda x: "+" if x >= 0 else ""
    print(f'Δ                       | 灰区准确度 {sign(delta_acc)}{delta_acc:+.1f}pp | '
          f'真进球召回 {sign(delta_recall)}{delta_recall:+.0f}pp | '
          f'误报识别 {sign(delta_mis)}{delta_mis:+.0f}pp | '
          f'极灰区 {sign(delta_ex)}{delta_ex:+.0f}pp')

agree = ds_win = qw_win = both_win = both_lose = 0
for i in common_ids:
    q = data[('qwen', '加密')][i]
    d = data[('ds', '加密')][i]
    q_ok = (q['vlm_verdict'] == 'goal') == (q['label'] == 1)
    d_ok = (d['vlm_verdict'] == 'goal') == (d['label'] == 1)
    if q['vlm_verdict'] == d['vlm_verdict']:
        agree += 1
    if q_ok and not d_ok:
        qw_win += 1
    if d_ok and not q_ok:
        ds_win += 1
    if q_ok and d_ok:
        both_win += 1
    if not q_ok and not d_ok:
        both_lose += 1
print()
print(f'加密版 两模型结论一致: {agree}/{len(common_ids)}')
print(f'  一致都判对: {both_win}，一致都判错: {both_lose}')
print(f'  分歧: qwen 对 ds 错 = {qw_win}；ds 对 qwen 错 = {ds_win}')

# 成本估算：22 帧 vs 16 帧
print()
print("成本估算（每个灰区事件）:")
print(f"  原版 qwen  16 帧: ~2 分 / 4 秒")
print(f"  加密 qwen  21 帧: ~3 分 / 8 秒（慢 2x，+1 分钱）")
print(f"  加密 deepseek 21 帧: ~2 分 / 3 秒（速度不变，token 单价不同）")
