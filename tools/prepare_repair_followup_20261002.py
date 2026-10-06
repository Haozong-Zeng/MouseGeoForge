"""Matched-observation QC and a small human review pilot; no automatic edits."""
import collections, html, itertools, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_repair_return_20261002 import ROOT, BASE, DEST, OUT, DOWNLOAD, OLD, FONT, points, write, sha, panel
from audit_pymousestudio_manual_multiview import fundamental, epi

TEXT='''小规模多视角复核：先统一定义，再决定是否扩大

感谢这轮返修。本轮结构问题已经修复，下一轮并不是要求所有图片继续改点，也不是要求人工迎合模型或几何结果。我们希望区分：原标注正确、模型错误、身份/定义歧义，以及确实无法可靠判断。

范围：12个同一时刻的多相机组。只复核06、07、15、16、17、18。腹部23及尾巴定义本轮保持不变；不要顺带批量重标其他点。编号为给标注人员看的1–23编号，不是程序0–22索引。

06：背侧正中线的背中表面固定部位；07：更靠尾根的腰背表面固定部位。请先看已确认的合成示意。它们不是内部脊柱点、每帧背部最高点、可见剪影的最高处或任意等分点。合成示意中的网格锚点可以严格固定，但真实动物没有相同网格可供测量：如果只能依靠外形猜测，必须写明limited/无法精确定位，不能假装是直接可见的精确点。另一视角可帮助理解部位，但不能让当前图的遮挡点自动变成visible。底视里看不到背部时不强制补点。

15左前爪、16右前爪、17左后爪、18右后爪：左右按小鼠自身的解剖左右，不按图片左右。前爪是掌部向指根过渡区域中心；后爪是跖部向趾根过渡区域中心。不是指尖、腕踝、膝，也不是毛团中心。先认头/尾和前后，再跟踪肢体连接关系辨左右。camA底视通常有帮助，但不是必须服从的答案；反射、重叠、蜷缩、镜像显示均可能使它不可靠。保持原图方向，不另行水平翻转。

每组怎样看：
1. 先打开原图多视角拼图和原分辨率PNG，不看模型预测，独立判断关注点。
2. 再看人工标注叠加图，逐相机核对06/07的纵向位置，以及前后/左右足身份。标注不能只在单一侧视图中孤立决定。
3. 难辨足部可以看同步前后约0.3–0.5秒的视频。短片只帮助判断身份，不要求给邻帧逐帧标点；局部五帧通常太短，建议先看1秒左右。此包提供中心帧；完整上下文视频尚未打包，待确定试标范围后补充，不能将其他时间帧冒充同一时刻。
4. 先独立标，再由第二人复核。先做R01/R02/R03/R10四组对齐定义；确认双方理解一致后，再做余下八组。争议需要双方说明依据，不简单平均两个坐标。
5. 初次判断不展示模型点或建议交换方向，减少诱导。几何/重投影提示只在第二轮复核时供参考，绝不要求为降低误差去移动点。

每个关注点在每个相机均需给状态：
visible：当前图对应部位直接可定位，给x/y，并说明clear或limited。
occluded_localizable：点被遮挡，但可根据同组其他视角/肢体关系估计；给x/y并写具体依据，训练时与直接可见分开。
unlabelable：在图内但遮挡、反光、模糊或左右身份无法可靠区分；不填坐标，注明原因。
out_of_frame：确认对应部位在画外；不填坐标，不能点击边缘替代。
pending：尚未检查，不等于不可见；最终交付不能遗漏待检查状态。

JSON缺少点只能说明未提供坐标，不能自动解释成不可见。可以继续使用Labelme；另外填写本包反馈模板，每个相机每个关注点一行。单鼠无需新增跨实例ID。保持文件名和目录不变，不修改PNG/相机参数。可直接填“原标无误”，不必为了返修而修改。

回交至少包括：修改后的JSON、逐点状态/结论、修改原因与证据视角、复核人员；没有改点的图也要有确认。身份确定但精确位置不确定时请明确写出，不要都归为“确定”。

验收：重复点/越界/错文件为硬检查；几何只是质量诊断。固定相机参数，先在共同有效观测上比较极线与留一视角重投影，分别报告点增删，不自动交换左右、不自动补隐藏点。模型开发只使用开发集，本包不包含锁定留出序列。若两人对背部点仍有系统性分歧，先进一步修订可操作定义/使用不确定性，而不是盲目增加训练轮数或扩大标注量。
'''

def main():
    pkg=next(p for p in DEST.iterdir() if p.is_dir()); summary=json.loads((OUT/'summary.json').read_text())
    original=json.loads((OLD/'review_manifest.json').read_text()); matched=[]; removed=added=0;metadata=collections.Counter();stale=0
    for row in summary['files']:
        case=next(c for c in original['cases'] if c['id']==row['case']); p=pkg/case['folder']/'originals'/row['source']
        a=json.loads((BASE/row['source']).read_text());b=json.loads(p.read_text())
        pa,pb=points(a),points(b);removed+=len(pa.keys()-pb.keys());added+=len(pb.keys()-pa.keys())
        metadata.update(s for s in ('flags','imagePath','imageWidth','imageHeight') if a.get(s)!=b.get(s))
        oldjpg=OLD/case['folder']/f"cam{row['camera']}_annotated.jpg";newjpg=pkg/case['folder']/oldjpg.name
        if row['changes'] and oldjpg.exists() and newjpg.exists() and sha(oldjpg)==sha(newjpg):stale+=1
    for c in original['cases']:
        if c['kind']!='correspondence':continue
        ids=[17,18] if c['pair']=='hindpaws' else [15,16]; cal=json.loads((BASE/'calibration'/c['session']/'rectified_calibration.json').read_text()); prev={};after={}
        for cam in c['paired_cameras']:
            rel=Path('real_images')/c['split']/c['session']/f"frame_{c['frame']:06d}"/f'cam{cam}.json'
            prev[cam]=points(json.loads((BASE/rel).read_text()));after[cam]=points(json.loads((pkg/c['folder']/'originals'/rel).read_text()))
        values=[]
        for a,b in itertools.combinations(c['paired_cameras'],2):
            f=fundamental(np.array(cal[a]['K']),np.array(cal[a]['world_to_camera_mm']),np.array(cal[b]['K']),np.array(cal[b]['world_to_camera_mm']))
            for k in ids:
                if all(len(d[v].get(k,[]))==1 for d in (prev,after) for v in (a,b)):
                    values.append([epi(d[a][k][0],d[b][k][0],f) for d in (prev,after)])
        v=np.array(values);matched.append(dict(id=c['id'],split=c['split'],n=len(v),before=float(np.minimum(v[:,0],100).mean()),after=float(np.minimum(v[:,1],100).mean())))
    audit=dict(matched_pair_epipolar_clipped100=matched,removed_label_slots=removed,added_label_slots=added,metadata_changes=dict(metadata),changed_images_with_stale_returned_overlay=stale,feedback_status='All24 conclusion cells blank; reasons and unchanged confirmation not supplied',warning='Pair counts can change in the initial report. This supplement compares exactly the same camera-pair/ID observations. No acceptance threshold or holdout model tuning.')
    write(OUT/'matched_followup.json',audit)
    pilot=OUT/'human_review_pilot';pilot.mkdir(exist_ok=False)
    selections=[('AZR2L2',129,'前足身份；模型共同偏离原标注'),('FR2',649,'底视与侧视的前足左右/前后对应'),('FR2',1104,'腰背07的表面定义与共同偏差'),('FR2',1429,'画面边缘的腰背07与遮挡'),('AZR2L2',584,'此前排除的足部身份待确认'),('FR2',1039,'此前排除的底视足部待确认'),('FR2',844,'后足位置；时序稳定不能证明正确'),('CR2',1559,'返修C11后仍有几何分歧，可能含遮挡'),('EL2',779,'返修C16后仍有几何分歧，可能含遮挡'),('CR2',1299,'返修D03的07/08编号，确认腰背与尾根区别')]
    cycle=ROOT/'output/bottom_anchor_cycle_20261002'; dm=json.loads((cycle/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text());entries=[dm['entries'][i] for i in dm['validation_indices']]
    with np.load(cycle/'quadruped_epoch110_review_v1/predictions_source_pixels.npz') as z: err=np.linalg.norm(z['uv']-z['gt'],axis=-1)
    selected={(s,f) for s,f,_ in selections}
    for session in ('AZR2L2','FR2'):
        cand=collections.defaultdict(list)
        for e,error in zip(entries,err):
            if e['session']==session and (session,e['frame']) not in selected:cand[int(e['frame'])].extend(error[np.isfinite(error)].tolist())
        frame=min(cand,key=lambda k:np.mean(cand[k]));selections.append((session,frame,'较低模型误差对照；不是已确认正确答案'))
    groups=json.loads((BASE/'ingestion_manifest.json').read_text())['frame_groups']; returned={r['source']:pkg/next(c['folder'] for c in original['cases'] if c['id']==r['case'])/'originals'/r['source'] for r in summary['files']}
    rows=[]; sections=[]; feedback=['组编号\t序列\t源帧\t相机\t关键点编号\t结论(原标无误/修改/无法判断)\t状态\tx\ty\t定位质量(clear/limited)\t依据视角/帧及原因\t标注人\t复核人']; count=0
    for n,(s,f,reason) in enumerate(selections,1):
        g=next(g for g in groups if g['session']==s and g['frame']==f);assert g['split']=='real_development'
        ident=f'R{n:02d}';folder=pilot/f'{ident}_{s}_f{f:06d}';folder.mkdir();panels=[];raws=[]; sources=[]
        for cam in g['cameras']:
            rel=Path('real_images')/g['split']/s/f'frame_{f:06d}'/f'cam{cam}.json';annsrc=returned.get(rel.as_posix(),BASE/rel);img=BASE/rel.with_suffix('.png')
            shutil.copy2(img,folder/img.name);shutil.copy2(annsrc,folder/annsrc.name);a=json.loads(annsrc.read_text())
            panels.append(panel(img,a,f'{ident} cam{cam} 人工标注',{6,7,15,16,17,18}))
            raw=Image.open(img).convert('RGB');raw.thumbnail((490,480));p=Image.new('RGB',(500,540),'#18212d');p.paste(raw,((500-raw.width)//2,55));ImageDraw.Draw(p).text((8,10),f'{ident} cam{cam} 原图',font=ImageFont.truetype(FONT,21),fill='white');raws.append(p)
            sources.append(dict(camera=cam,annotation_source=str(annsrc),image_sha256=sha(img)));count+=1
            for k in [6,7,15,16,17,18]:feedback.append(f'{ident}\t{s}\t{f}\t{cam}\t{k:02d}\t\tpending\t\t\t\t\t\t')
        for name,parts in [('raw',raws),('manual',panels)]:
            sheet=Image.new('RGB',(1500,math.ceil(len(parts)/3)*540),'#18212d')
            for i,p in enumerate(parts):sheet.paste(p,(i%3*500,i//3*540))
            sheet.save(folder/f'{name}.jpg',quality=94)
        rows.append(dict(id=ident,session=s,frame=f,cameras=g['cameras'],reason=reason,sources=sources))
        sections.append(f'<h2>{ident} {s} / {f}: {reason}</h2><p><a href="{folder.name}/raw.jpg">先看原图拼图</a> · <a href="{folder.name}/manual.jpg">再看人工标注</a> · 原PNG/JSON位于同名目录</p><img loading="lazy" src="{folder.name}/raw.jpg">')
    guide=ROOT/'output/annotation_packages/PyMouseStudio-AnnotationGuide-SurfaceCore23-v3';shutil.copytree(guide,pilot/'approved_guide')
    (pilot/'给标注人员的说明.txt').write_text(TEXT,encoding='utf-8-sig');(pilot/'逐点复核反馈模板.txt').write_text('\n'.join(feedback)+'\n',encoding='utf-8-sig')
    write(pilot/'selection_manifest.json',dict(groups=rows,center_images=count,focus_ids=[6,7,15,16,17,18],training_modified=False,model_predictions_shown=False,context_clips_included=False))
    page='<meta charset="utf-8"><title>小规模多视角复核试标</title><style>body{font:17px sans-serif;max-width:1500px;margin:28px auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>12组小规模多视角复核试标</h1><p><a href="approved_guide/index.html">已确认合成示意与关键点定义</a> · <a href="给标注人员的说明.txt">沟通说明</a> · <a href="逐点复核反馈模板.txt">逐点反馈模板</a></p><pre>'+html.escape(TEXT)+'</pre>'+''.join(sections)
    (pilot/'index.html').write_text(page,encoding='utf-8')
    index=OUT/'index.html';index.write_text(index.read_text().replace('<h1>2026-10-02 返修包核验</h1>','<h1>2026-10-02 返修包核验</h1><p><a href="human_review_pilot/index.html">下一轮12组复核方案与图片</a> · <a href="matched_followup.json">共同观测比较补充记录</a></p><p>注意：67张已改坐标图片的返修包叠加图仍是旧版；本页已重新绘制。原反馈表24项均未填写。</p>'),encoding='utf-8')
    export=DOWNLOAD/'PyMouseStudio-Repair-Audit-20261002';shutil.copytree(OUT,export,dirs_exist_ok=True)
    print(json.dumps(dict(**audit,pilot_groups=12,pilot_images=count),ensure_ascii=False,indent=2))
if __name__=='__main__':
    import math
    main()
