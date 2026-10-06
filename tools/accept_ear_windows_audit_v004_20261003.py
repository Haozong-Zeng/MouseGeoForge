import datetime as dt,hashlib,json,shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_ear_cycle_20261003';plan=json.loads((base/'plan.json').read_text());audit=json.loads((base/'windows_audit_v004/audit.json').read_text());assert audit['complete']
delivery=Path(plan['project_directory'])/'DELIVERY';windows=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-EarStudy-20261003/DELIVERY')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p,s in plan['sources'].items():assert sha(Path(s))==plan['source_sha256'][p]
for r in audit['records']:
    assert r['packed_images']==132 and r['image_file_dependencies']==132 and not r['missing_images'] and r['external_libraries']==0
    assert len(r['hair_settings'])==9 and all(x['viewport_disabled'] and not x['render_disabled'] for x in r['hair_settings'])
shutil.copy2(base/'windows_audit_v004/audit.json',delivery/'evidence/windows_copy_audit.json')
p=delivery/'index.html';page=p.read_text();assert 'id="windows-audit"' not in page
section='<section id="windows-audit"><h2>Windows下载副本复核</h2><p>直接重开下载目录中的三个工程，确认每个工程132项图像文件依赖全部内嵌、无外链库、无缺失图像；9个毛发/胡须对象GUI隐藏且渲染开启。耳材质基础灰度0.60、默认F相机与静态帧均正确，随附脚本的OptiX设备检查通过，检查过程中.blend校验值未变。</p><p><a href="evidence/windows_copy_audit.json">独立副本检查记录</a></p></section>'
p.write_text(page.replace('</main>',section+'</main>'),encoding='utf-8')
manifest=json.loads((delivery/'package_manifest.json').read_text());manifest['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat();manifest['files']=[dict(path=p.relative_to(delivery).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name!='package_manifest.json'];manifest['payload_bytes']=sum(x['bytes'] for x in manifest['files']);(delivery/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for relative in ['index.html','package_manifest.json','evidence/windows_copy_audit.json']:shutil.copy2(delivery/relative,windows/relative)
for entry in manifest['files']:assert sha(windows/entry['path'])==entry['sha256'],entry['path']
class Links(HTMLParser):
    def __init__(self):super().__init__();self.items=[]
    def handle_starttag(self,tag,attrs):self.items.extend(v for k,v in attrs if k in ('href','src'))
links=Links();links.feed((windows/'index.html').read_text())
for l in links.items:
    if not urlparse(l).scheme:assert (windows/unquote(l)).is_file(),l
verification=dict(utc=manifest['updated_utc'],windows_reopened=True,packed_images_each=132,missing_images=0,source_three_sha_unchanged=True,payload_files=len(manifest['files']),payload_bytes=manifest['payload_bytes'],all_windows_hashes_match=True,links_checked=len(links.items))
(base/'final_integrity_v004.json').write_text(json.dumps(verification,indent=2))
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));assert state['active_pid'] is None
state.update(stage='ear_cycle_verified_ready_for_shutdown',active_job=None,updated_utc=manifest['updated_utc'],final_integrity=str(base/'final_integrity_v004.json'),next_action='ALL WORK COMPLETE except scheduledclosure. v004 Windowsactual3blends reopened/dryrunOptiX, all132fileimagespacked,0missing,0libraries,9hairGUIoff/renderon,ear.60,frames31/111/216 correct,shaunchanged. Report+evidencecopiedboth,39payloadfiles~924MB allSHA/39linksverified,original3deliverySHAunchanged. No renders/newexperimentsneeded. NEXT heartbeat08:50 (deadline08:58) do briefownedPID/GPUlockcheck, readopenai-docsskill officialautomationdocs ifneeded; pausepymousestudio viaapptool preservingfields (readTOML withtomllib),confirmPAUSEDreadback,recordcycle_final_statusandclosedstate,newreportclosedbanner+manifestcopyverify. Do not modifyoldRealCaptureDELIVERY. Final useranswerChinese reportlink,ear.12->.60only,3staticblends12images,multi-viewgeometryunchanged,ears shape/thicknessstillnotreal. BrowserlocalfileURLblockedno workaround. User requestfulfilled localearappearanceimprovement withlimitations;cycleendnotphotorealisticfullmouseclaim.')
(base/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));print(json.dumps(verification,indent=2))
