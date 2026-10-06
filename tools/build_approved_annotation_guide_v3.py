"""Package user-approved surface demos with explicit human label contracts."""
import csv
import hashlib
import html
import json
import math
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import package_surface_anchor_annotation_demo as demo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/annotation_packages/PyMouseStudio-AnnotationGuide-SurfaceCore23-v3'
VERSION='pymousestudio_manual_surface_core23_v3'
MANDATORY={0,5,6,7,13,14,15,16,17,22}
RENAMED={5:('dorsal_thorax_surface','背中表面中心'),6:('dorsal_lumbar_surface','腰背表面中心'),22:('ventral_abdomen_surface','腹面表面中心')}
WITH_XY={'visible','occluded_localizable'}
WITHOUT_XY={'unlabelable','out_of_frame','deferred_definition','pending'}


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def font(n):return ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',n)


def validate_state(data,completed=False):
    assert data['definition_version']==VERSION
    for instance in data['instances']:
        assert len(instance['keypoints'])==23
        assert [p['id'] for p in instance['keypoints']]==list(range(23))
        for p in instance['keypoints']:
            state=p['state'];assert state in WITH_XY|WITHOUT_XY
            if state in WITH_XY:
                assert isinstance(p['x'],(float,int)) and isinstance(p['y'],(float,int))
                assert math.isfinite(p['x']) and math.isfinite(p['y'])
                assert 0<=p['x']<data['image_width'] and 0<=p['y']<data['image_height']
                assert p['evidence_quality'] in {'clear','limited'}
                if state=='occluded_localizable':assert bool(p['reason'])
            else:assert p['x'] is None and p['y'] is None
            if completed and p['id'] in MANDATORY:assert state not in {'pending','deferred_definition'}


def main():
    demo.main(['--approved-surface','--output',str(OUT),'--skip-zip'])
    templates=OUT/'templates';templates.mkdir()
    native=OUT/'labelme_example';native.mkdir()
    old=json.loads((ROOT/'dataset/PyMouseStudio/annotation_pack_v2/schema.json').read_text())['keypoints']
    descriptions={
        0:'鼻尖最前端可见中心；不是额头、吻部中段、胡须或头部装置。',
        5:'按已确认示意，背侧正中线的背中表面固定部位；以静止网格Spine3纵向截面为基准。不是每帧背部最高点，不是内部脊柱。',
        6:'按已确认示意，背侧正中线更靠尾根方向的腰背表面固定部位；以静止网格Spine1纵向截面为基准。不逐帧追随剪影最高点。',
        7:'尾巴从躯干伸出的位置中心；不能把较远处尾段或可见段起点当尾根。',
        13:'真实解剖尾尖；不是画布边缘、遮挡边缘、可见尾段终点。真正尾尖在画外时留空并标out_of_frame。',
        14:'小鼠自身左前爪：掌部向指根过渡区域的中心；不是最长指尖，也不是腕。',
        15:'小鼠自身右前爪：掌部向指根过渡区域的中心；不是最长指尖，也不是腕。',
        16:'小鼠自身左后爪：跖部向趾根过渡区域的中心；不是最长趾尖，也不是踝或膝。',
        17:'小鼠自身右后爪：跖部向趾根过渡区域的中心；不是最长趾尖，也不是踝或膝。',
        22:'按已确认示意，腹侧正中线的腹面固定部位，与06的静止纵向截面对齐。不是背部重复点，不是侧面肚子剪影最低点；看不到不能强行点击。'}
    deferred={1:'旧头部内部骨骼中心，不能用植入装置中心或眼中点代替。',
        2:'旧左耳骨起点与耳尖/耳附着中心尚未统一。',3:'旧右耳骨起点与耳尖/耳附着中心尚未统一。',
        4:'旧颈部内部骨骼中心尚未转为明确表面定义。',
        18:'旧left_elbow标签引用Wrist骨起点，尚不作为已确认的真实肘或腕定义。',
        19:'旧right_elbow标签引用Wrist骨起点，尚不作为已确认的真实肘或腕定义。',
        20:'内部左膝中心与可观察皮肤标志尚未统一；不能用后足近端/踝代替。',
        21:'内部右膝中心与可观察皮肤标志尚未统一；不能用后足近端/踝代替。'}
    for k in range(8,13):deferred[k]='旧尾部中间骨点；不是已确认的等弧长点。暂不手工填此槽，可另外画可见尾中线。'
    definitions=[]
    for k,row in enumerate(old):
        name,cn=RENAMED.get(k,(row['name'],row['chinese'].replace('（暂缓）','')))
        definitions.append(dict(id=k,display_number=k+1,name=name,labelme_label=f'{k+1:02d}{name}',chinese=cn,
            requirement='required_state' if k in MANDATORY else 'deferred_definition',
            definition=descriptions.get(k,deferred.get(k,'')),
            legacy_name=row['name'],definition_version=VERSION,
            old_label_auto_compatible=k not in RENAMED,manual_exact_rig_equivalence_confirmed=False))
    write(OUT/'keypoint_definitions.json',definitions)
    with (OUT/'keypoint_definitions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(definitions[0]));writer.writeheader();writer.writerows(definitions)
    mapping=dict(definition_version=VERSION,approved_surface_ids=[6,7,23],
        approval_date='2026-09-13',approval_basis='user visually reviewed synthetic surface demo and accepted',
        annotation_required_ids=[k+1 for k in sorted(MANDATORY)],
        legacy_aliases=[dict(old=f'{k+1:02d}{old[k]["name"]}',new=f'{k+1:02d}{RENAMED[k][0]}',
                            requires_coordinate_review=True,automatic_rename_is_not_conversion=True) for k in RENAMED],
        changes_training_data=False)
    write(OUT/'definition_version.json',mapping)
    # The geometry recipe stays immutable; this companion records approval.
    anchors=json.loads((OUT/'rest_surface_anchors.json').read_text())
    anchors['approval']=dict(status='user_approved_surface_locations',date='2026-09-13',applies_to_display_ids=[6,7,23])
    write(OUT/'rest_surface_anchors.json',anchors)
    states=[
        ('visible','直接可见','关键点对应的部位在当前图可辨认，可以直接定位。肢体其他部分被挡不影响这个点的状态。','必须填x/y'),
        ('occluded_localizable','被挡住，但可推测','关键点本身不可见，依据姿态或同组其他视角能合理估计；必须备注依据，不能标成visible。','填x/y，并写推测依据'),
        ('unlabelable','无法可靠定位','可能在图内，但被遮挡、模糊、反光覆盖或左右无法确定；宁可留空，不凭模板猜。','x/y为null'),
        ('out_of_frame','真正的点在画外','确认解剖点在画布外；不可点击图像边缘替代它。尾巴被截断时特别注意。','x/y为null'),
        ('deferred_definition','定义暂缓','仅用于定义表中暂缓的13点；不等于身体没有这个器官。','x/y为null'),
        ('pending','尚未处理','只允许在工作中的模板存在；最终回交的10个要求项不能保留pending。','x/y为null')]
    write(OUT/'visibility_contract.json',dict(definition_version=VERSION,
        coordinate_space='same rectified PNG; 0-based continuous pixels, x right, y down',
        states=[dict(state=s,chinese=c,definition=d,coordinates=x) for s,c,d,x in states],
        states_with_coordinates=sorted(WITH_XY),states_without_coordinates=sorted(WITHOUT_XY),
        evidence_quality=dict(clear='部位清楚',limited='局部遮挡/模糊，但点仍可直接定位或推测；不能改变state的含义',not_applicable='无坐标时'),
        instance_and_keypoint_visibility_are_distinct=True,percentage_visibility_required=False,
        absent_point_is_not_absent_animal=True,missing_coordinates_must_not_be_zero_filled=True,
        completion_requires_state_for_all_23_slots=True,required_state_ids=sorted(k+1 for k in MANDATORY)))
    template=dict(definition_version=VERSION,image=None,image_sha256=None,image_width=1080,image_height=1200,
        session=None,source_frame=None,camera=None,split='preserve_original_manifest',train_allowed=False,
        annotator=None,reviewer=None,annotation_status='pending',
        instances=[dict(instance_id=1,identity_verified=False,anatomical_lr_verified=False,bbox_body_xyxy=None,
            bbox_extent='visible',mask_extent='visible',mask_includes_tail=True,
            keypoints=[dict(id=r['id'],name=r['name'],labelme_label=r['labelme_label'],x=None,y=None,
                state='pending' if r['id'] in MANDATORY else 'deferred_definition',evidence_quality='not_applicable',reason=None) for r in definitions])])
    write(templates/'IMAGE.annotation_state.template.json',template)
    (templates/'labelme_labels.txt').write_text('\n'.join(['mouse']+[r['labelme_label'] for r in definitions])+'\n',encoding='utf-8')
    # Real Labelme-native illustration using synthetic geometry, clearly NOT human GT.
    stem='WalkAround_frame000031_view06_bottom'
    image_src=OUT/'synthetic_examples'/f'{stem}_clean.png'
    shutil.copy2(image_src,native/'synthetic_bottom.png')
    example=json.loads((OUT/'synthetic_examples'/f'{stem}_gt.json').read_text())
    p=example['keypoints'][22];label=definitions[22]['labelme_label']
    labelme=dict(version='5.5.0',flags={'synthetic_example_only':True},shapes=[dict(label=label,points=[p['uv']],
        group_id=1,description='教学用synthetic表面点；不是人工真值。',shape_type='point',
        flags={'visible':True,'occluded_localizable':False},mask=None)],imagePath='synthetic_bottom.png',imageData=None,imageHeight=512,imageWidth=512)
    write(native/'synthetic_bottom.json',labelme)
    sample=json.loads(json.dumps(template));sample.update(image='synthetic_bottom.png',image_width=512,image_height=512,
        session='SYNTHETIC_EXAMPLE_ONLY',source_frame=31,camera='view06',annotation_status='example_incomplete',
        annotation_source='synthetic_geometry_example_not_human_gt',image_sha256=hashlib.sha256(image_src.read_bytes()).hexdigest())
    q=sample['instances'][0]['keypoints'][22];q.update(x=p['uv'][0],y=p['uv'][1],state='visible',evidence_quality='limited',reason='教学演示：底视腹面点，状态不代表真实数据人工审核。')
    write(native/'synthetic_bottom.annotation_state.json',sample)
    validate_state(template);validate_state(sample)
    for state in WITH_XY|WITHOUT_XY:
        trial=json.loads(json.dumps(template));q=trial['instances'][0]['keypoints'][0];q['state']=state
        if state in WITH_XY:q.update(x=123.5,y=456.0,evidence_quality='limited',reason='仅为格式验证的推测说明')
        validate_state(trial)
    # Machine-check the invalid combinations and incomplete completion gate.
    rejected=0
    for kind in range(3):
        bad=json.loads(json.dumps(template));q=bad['instances'][0]['keypoints'][0]
        if kind==0:q.update(state='visible',x=None,y=None)
        if kind==1:q.update(state='out_of_frame',x=0,y=0)
        try:validate_state(bad,completed=kind==2)
        except AssertionError:rejected+=1
    assert rejected==3
    quick='''# 先读：新版标注说明 v3

负责人已于2026-09-13目视确认06背中、07腰背、23腹面三个固定表面部位。本包替代旧demo中这三点的定义与相关可见性说明；不替换已交付的待标真实图片。

## 阅读顺序

1. 01_rest_surface_definition.jpg：从默认静止网格的俯视/底视确定三个表面部位。
2. keypoint_definitions.csv 或 01_关键点定义.md：查编号、名称和是否需要本轮标注。
3. 02_可见程度与遮挡要求.md、visibility_quick_reference.png：每点分别填状态。
4. 03_Labelme回交格式与检查清单.md：按示例回交原生JSON加逐点状态文件。
5. synthetic_examples：9组走路/洗脸/攀墙的侧、底、顶视完整编号示例。

## 本轮必须逐项处理的10个点

01鼻尖、06背中表面、07腰背表面、08尾根、14真实尾尖、15/16左右前爪、17/18左右后爪、23腹面表面。

“必须处理”指必须给出状态，**不是每图必须点击10个坐标**。其余13点仍为deferred_definition，不能因为这次批准了背腹位置就强制凑齐23点。

可见性针对关键点本身，不是整只鼠、整条腿或整个mask的可见比例。不要求主观填写0–100%。同组另一相机可见，不代表当前图可见。

所有真实图仍使用原包1080×1200已矫正PNG，左上原点、x右y下；不得二次畸变矫正、镜像、旋转、裁剪或缩放后直接回交坐标。左右按小鼠自身解剖左右。

## 这次改名

06spine_center → 06dorsal_thorax_surface；07lumbar_spine → 07dorsal_lumbar_surface；23abdomen_center → 23ventral_abdomen_surface。

保留编号方便对应，但语义已变；旧坐标不能只改名字就当新版。此前AZL3第0帧需要复核这三点、尾尖与可见性状态，不在本次自动修改。

本包示意坐标不是待标真实图的预标注。旧训练数据、相机及模型尚未迁移；train_allowed=false是审核/训练隔离，不妨碍人工按新版规则工作。
'''
    (OUT/'00_先读我.md').write_text(quick,encoding='utf-8')
    rows=['# 关键点定义 v3','', '编号1–23；文件id为0–22。`required_state`为必须处理状态；`deferred_definition`为暂缓，坐标留空。','',
          '|编号|Labelme点名|中文|要求|定位说明|','|---|---|---|---|---|']
    for r in definitions:rows.append(f"|{r['display_number']:02d}|{r['labelme_label']}|{r['chinese']}|{'处理状态' if r['id'] in MANDATORY else '暂缓'}|{r['definition']}|")
    rows+=['','## 表面定义补充','',
        '06与23在默认静止姿态处于相同纵向截面的背/腹两面。动画中追随同一网格区域，而不是重新找最高/最低点。毛发覆盖下标相同身体部位，不找某一根毛发尖端。',
        '身体被旋转或抬起后，当前相机可能看不到这个部位；必须如实给遮挡/无法定位状态。不能把隐藏腹面点放到可见侧面轮廓上。',
        '人工真实观察点与合成内部骨骼并非全部精确同义。耳尖、眼中心、腕/肘/踝/膝等额外观察可另记extra_observations，不能静默替换暂缓槽位。',
        '尾中线可用linestrip按可见片段绘制；每段同一实例group_id，遮挡处分段，不跨遮挡拉直线。派生等弧长点不是已确认的跨视图同一3D物理点。']
    (OUT/'01_关键点定义.md').write_text('\n'.join(rows),encoding='utf-8')
    visibility=['# 可见程度与遮挡要求','', '|state|中文|如何判断|坐标|','|---|---|---|---|']
    for s,c,d,x in states:visibility.append(f'|{s}|{c}|{d}|{x}|')
    visibility+=['','## 部分可见怎么标','',
        '- 只露出部分爪子，但掌/指根中心本身可辨认：visible；evidence_quality可记limited。',
        '- 爪子轮廓能看到，但规定的掌中心被挡住，只能推测：occluded_localizable，并说明依据。',
        '- 部位被挡住且没有可靠定位依据，或左右无法确定：unlabelable，不强行点击。',
        '- 模糊/毛发覆盖但部位仍能定位：按直接证据是否足够选择visible/limited；仅靠姿势推断则occluded_localizable，无法稳定估计则unlabelable。',
        '- 只有另一个视角看得到腹部：当前视角不得标visible。参考别的视图只能辅助推测，不复制其二维坐标。',
        '- 真尾尖在画外：out_of_frame。可见尾段终点可另存extra_observations，不能占用14号。',
        '- 背中/腹面点在某个二维投影位置很接近，不代表同一3D点；不可为避免重合而沿轮廓挪动。',
        '- 反射里的点不算直接观察到真实鼠；头部装置不能代替头或鼻的解剖标志。',
        '', '## 坐标与质量字段','',
        '`visible`与`occluded_localizable`必须有有限x/y，范围0≤x<width、0≤y<height。其余状态x/y均为null，不用(0,0)、(-1,-1)或图边缘占位。',
        '`evidence_quality`仅用clear/limited/not_applicable，不替代state。无坐标用not_applicable；推测点必须写reason。可另外记录像素不确定范围，但这不是已校准的模型置信度。',
        '完成时所有23槽都须有状态：10个要求项不能pending/deferred；13个暂缓项保留deferred_definition。缺点不是无鼠，缺标不能产生空目标负样本。',
        '', '## synthetic图例的限制','',
        '新版06/07/23实心/空心仅由裸鼠网格自遮挡计算；不包含毛发、装置和场景遮挡，更不是人工标注状态。其余旧点空心仅作编号参考。标注员必须根据真实图重新判断。']
    (OUT/'02_可见程度与遮挡要求.md').write_text('\n'.join(visibility),encoding='utf-8')
    delivery='''# Labelme回交格式与检查清单

## 文件组织

保留原 real_images/<split>/<session>/frame_<源序号>/camX.png 路径。建议同目录放camX.json（Labelme原生）及camX.annotation_state.json（本项目状态表）。必须保留session/帧/相机/split，不能只交多个同名camA.json而丢失来源。

Labelme原生文件可直接用软件编辑；annotation_state文件是本项目补充格式，**不能当Labelme原生文件打开**。纯原生Labelme不足以无歧义表达“没有坐标的点为何缺失”，因此状态表必须一起回交；也可使用能保留等价字段的平台原生导出，但不要只交坐标。

## 如何填写

1. 点label按keypoint_definitions.csv，保留两位编号和新版名称。所有真实鼠点及mask都写group_id，同一实例所有形状使用同一整数；单鼠也填1，不留null。
2. 有坐标的点在Labelme中使用shape_type=point；flags可设visible=true或occluded_localizable=true，二者互斥。软件不便填写flags时，至少完整填逐点状态表；若两处同时填写，必须一致。
3. 无坐标点不创建Labelme占位shape，只在状态表写unlabelable/out_of_frame/deferred_definition。不要画到(0,0)，也不要直接删掉状态项。
4. state文件从templates/IMAGE.annotation_state.template.json复制，改image、尺寸、session、source_frame、camera、split、annotator等。可定位点的x/y须与Labelme点一致；推测点reason不可空。identity_verified与anatomical_lr_verified只在确实核对后置true。
5. 完成时annotation_status改为annotated，填写标注员及复核者。暂未复核的reviewer可null，但不能声称已经复核。保持train_allowed=false，是否导入训练由负责人处理。

labelme_example/synthetic_bottom.json是可打开的Labelme教学示例，旁边有PNG和对应状态表。只示范一个23号点，其余待处理，是**未完成示例**，不能拿来充当真实回标。

## 框和分割

- 建议body-only框包含可见头耳躯干四肢，不含尾巴、倒影、影子和装置；rectangle标签body_bbox。可另存full_animal_bbox，但不能混称。
- 已绘制的mouse polygon按可见实体审核，可包含尾巴，标明mask_includes_tail=true。反射/装置不算鼠；不要为了连成整体跨遮挡画桥。
- 同鼠多个可见片段可用多个polygon、同group_id。尾环内部真实背景应保留；复杂孔洞可另交原尺寸二值mask并记录路径，不用自交多边形勉强表达。
- 不对不可见身体补“完整轮廓”；若另做amodal标注，必须另层另名，不能混入visible mask。

## 回交前检查

- 图片未二次矫正、裁剪、缩放或镜像；尺寸与JSON一致。
- 01/06/07/08/14/15/16/17/18/23均已处理状态；其余暂缓项保留null。
- 背部与腹部不重复点击同一背部位置；尾尖不点击画布/遮挡截断处。
- 左右按小鼠自身；同时间组各相机独立判断可见性。
- 所有点、mask和框有实例ID；单鼠也不漏。
- 无坐标点没有伪坐标；Labelme与状态表坐标/名称/状态一致。
- 保留原来源分组与holdout；不因标注完成而改训练许可。
'''
    (OUT/'03_Labelme回交格式与检查清单.md').write_text(delivery,encoding='utf-8')
    # A compact scientific state table, not generated illustrative anatomy.
    card=Image.new('RGB',(1500,1130),'#14202b');d=ImageDraw.Draw(card)
    d.text((35,22),'关键点可见程度：先判断点本身，再决定是否填写坐标',font=font(34),fill='white')
    d.text((35,80),'部分身体可见 ≠ 关键点可见；另一相机看得到 ≠ 当前相机可见',font=font(26),fill='#71d5fc')
    for i,(s,c,definition,coords) in enumerate(states):
        y=150+i*137
        d.rounded_rectangle((25,y,1475,y+123),radius=12,fill='#203444')
        d.text((45,y+12),c,font=font(28),fill='#ffd48b')
        d.text((440,y+15),s,font=font(26),fill='white')
        d.text((1110,y+15),coords,font=font(20),fill='#99e3ba')
        # Wrap by measured pixel width.
        line='';lines=[]
        for char in definition:
            if d.textlength(line+char,font=font(23))>1370:lines.append(line);line=''
            line+=char
        lines.append(line)
        for j,line in enumerate(lines):d.text((45,y+56+j*29),line,font=font(23),fill='#e2e8ee')
    d.text((35,1000),'10个要求项：01 / 06 / 07 / 08 / 14 / 15 / 16 / 17 / 18 / 23',font=font(28),fill='white')
    d.text((35,1050),'必须处理状态，不是强制点击坐标。其他13点定义暂缓；左右按小鼠自身。',font=font(27),fill='#ffd48b')
    card.save(OUT/'visibility_quick_reference.png')
    # Replace review-only landing text; preserve scientific image content.
    gallery=(OUT/'index.html').read_text()
    gallery=gallery.replace('表面锚点示意 v3','标注说明 v3：表面位置已确认').replace('背部 / 腹部：固定网格表面定义（待审核）','标注说明 v3：06/07/23表面位置已确认')
    navigation='<section><h2>标注员先读</h2><p>这次批准06/07/23；本轮10个点要求处理状态，其余13点仍暂缓。</p>'
    for name in ['00_先读我.md','01_关键点定义.md','02_可见程度与遮挡要求.md','03_Labelme回交格式与检查清单.md']:
        body=(OUT/name).read_text();page=Path(name).with_suffix('.html').name
        (OUT/page).write_text('<!doctype html><meta charset="utf-8"><title>'+name+'</title><style>body{max-width:1100px;margin:32px auto;font:18px/1.7 sans-serif;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}a{color:#1473b8}</style><a href="index.html">返回首页</a><pre>'+html.escape(body)+'</pre>',encoding='utf-8')
        navigation+=f'<p><a href="{page}">{name[:-3]}</a></p>'
    navigation+='<p><a href="keypoint_definitions.csv">关键点定义CSV</a> · <a href="templates/IMAGE.annotation_state.template.json">逐点状态模板</a></p><img style="max-width:100%" src="visibility_quick_reference.png"></section>'
    gallery=gallery.replace('<h1>',navigation+'<h1>',1)
    (OUT/'index.html').write_text(gallery,encoding='utf-8')
    validation=dict(status='passed',required_points=10,deferred_points=13,definition_count=len(definitions),
        valid_state_cases=6,invalid_state_cases_rejected=rejected,template_and_native_example_consistent=True,
        approved_surface_ids=[6,7,23],training_started=False)
    assert len(definitions)==23 and len(MANDATORY)==10
    assert labelme['shapes'][0]['label']==sample['instances'][0]['keypoints'][22]['labelme_label']
    assert labelme['shapes'][0]['points'][0]==[sample['instances'][0]['keypoints'][22]['x'],sample['instances'][0]['keypoints'][22]['y']]
    for f in OUT.rglob('*.png'):Image.open(f).verify()
    for f in OUT.rglob('*.jpg'):Image.open(f).verify()
    write(OUT/'validation.json',validation)
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'}
    write(OUT/'checksums.json',hashes)
    for name,digest in hashes.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(hashes)+1,validation=validation)
    write(archive.with_suffix('.receipt.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
