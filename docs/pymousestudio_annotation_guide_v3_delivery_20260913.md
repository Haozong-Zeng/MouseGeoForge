# PyMouseStudio标注说明v3交付

用户已目视确认背中/腰背/腹面固定表面示意，2026-09-13制作正式说明补充包。

- ZIP：`output/annotation_packages/PyMouseStudio-AnnotationGuide-SurfaceCore23-v3.zip`。
- 大小：20,191,690 bytes（约20.2MB），93个文件。
- SHA256：`8deb2b3b0c6609c67d8055e44ca5ab98be8c5449e61aac17a7f2159ac9712ed7`。
- 包内首页`index.html`，阅读入口`00_先读我.md`及同名HTML，离线可用。
- 9组合成多视图示例、3动作前后对照、静止网格定位、12张灰模图及3个轻量Blender检查文件。
- 23槽位CSV/JSON定义；10个点要求处理状态，13个未确认语义点继续暂缓。
- visible / occluded_localizable / unlabelable / out_of_frame / deferred_definition / pending，明确何时允许坐标、部分可见如何判断、最终不能保留哪些pending项。
- Labelme教学示例是真实原生格式；独立annotation_state模板是本项目补充格式，不能当Labelme原生文件导入。
- 图面复核：可见程度速查卡和已批准静止定位图可读、没有标签重叠；图像全解码、文件哈希和ZIP CRC通过；6类合法状态检查与3类非法状态拒绝测试通过。
- 旧审核包、外置硬盘真实图片、已回交标注、生产合成数据与模型未覆盖或修改。新版说明不自动批准训练。

可复现生成器：`tools/build_approved_annotation_guide_v3.py`；复用示意图生成器的`--approved-surface`模式。
