from __future__ import annotations

import html
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image


EMU_PER_IN = 914400
SLIDE_W = 13.333333 * EMU_PER_IN
SLIDE_H = 7.5 * EMU_PER_IN


def emu(inches: float) -> int:
    return int(inches * EMU_PER_IN)


def xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def text_runs(text: str, size=24, color="1F2937", bold=False):
    lines = text.split("\n")
    parts = []
    for line in lines:
        line = xml_escape(line)
        b = ' b="1"' if bold else ""
        parts.append(
            f"""
            <a:p>
              <a:r>
                <a:rPr lang="zh-CN" sz="{size * 100}"{b}>
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Microsoft YaHei"/>
                  <a:ea typeface="Microsoft YaHei"/>
                </a:rPr>
                <a:t>{line}</a:t>
              </a:r>
            </a:p>
            """
        )
    return "\n".join(parts)


def shape_text(shape_id, name, x, y, w, h, text, size=24, color="1F2937", bold=False, fill=None):
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:noFill/></a:ln>'
        if fill
        else '<a:noFill/><a:ln><a:noFill/></a:ln>'
    )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="{xml_escape(name)}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        {fill_xml}
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="120000" tIns="80000" rIns="120000" bIns="80000"/>
        <a:lstStyle/>
        {text_runs(text, size=size, color=color, bold=bold)}
      </p:txBody>
    </p:sp>
    """


def rect(shape_id, x, y, w, h, fill="EEF2FF", line="CBD5E1"):
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="Rectangle {shape_id}"/>
        <p:cNvSpPr/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="12000"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
    </p:sp>
    """


def image_pic(shape_id, name, rid, x, y, w, h):
    return f"""
    <p:pic>
      <p:nvPicPr>
        <p:cNvPr id="{shape_id}" name="{xml_escape(name)}"/>
        <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
        <p:nvPr/>
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="{rid}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </p:spPr>
    </p:pic>
    """


def fit_image(path: Path, x_in, y_in, max_w_in, max_h_in):
    with Image.open(path) as img:
        w_px, h_px = img.size
    ratio = min(max_w_in / w_px, max_h_in / h_px)
    w_in = w_px * ratio
    h_in = h_px * ratio
    x = x_in + (max_w_in - w_in) / 2
    y = y_in + (max_h_in - h_in) / 2
    return emu(x), emu(y), emu(w_in), emu(h_in)


class Slide:
    def __init__(self, title, subtitle=None):
        self.title = title
        self.subtitle = subtitle
        self.elements = []
        self.images = []
        self.shape_id = 10

    def add_text(self, x, y, w, h, text, size=22, color="1F2937", bold=False, fill=None):
        self.shape_id += 1
        self.elements.append(shape_text(self.shape_id, "Text", emu(x), emu(y), emu(w), emu(h), text, size, color, bold, fill))

    def add_box(self, x, y, w, h, text, size=20, fill="EEF2FF", color="111827", bold=False):
        self.shape_id += 1
        self.elements.append(rect(self.shape_id, emu(x), emu(y), emu(w), emu(h), fill=fill))
        self.shape_id += 1
        self.elements.append(shape_text(self.shape_id, "BoxText", emu(x), emu(y), emu(w), emu(h), text, size, color, bold))

    def add_image(self, path, x, y, max_w, max_h):
        path = Path(path)
        if not path.exists():
            self.add_text(x, y, max_w, max_h, f"Image missing:\n{path}", size=16, color="B91C1C")
            return
        rid = f"rId{len(self.images) + 1}"
        self.images.append((rid, path))
        ix, iy, iw, ih = fit_image(path, x, y, max_w, max_h)
        self.shape_id += 1
        self.elements.append(image_pic(self.shape_id, path.name, rid, ix, iy, iw, ih))

    def xml(self):
        title = shape_text(2, "Title", emu(0.45), emu(0.22), emu(12.4), emu(0.55), self.title, size=25, color="111827", bold=True)
        bar = f"""
        <p:sp>
          <p:nvSpPr><p:cNvPr id="3" name="Accent"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
          <p:spPr>
            <a:xfrm><a:off x="{emu(0.45)}" y="{emu(0.85)}"/><a:ext cx="{emu(12.4)}" cy="{emu(0.03)}"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="2563EB"/></a:solidFill>
            <a:ln><a:noFill/></a:ln>
          </p:spPr>
        </p:sp>
        """
        subtitle = ""
        if self.subtitle:
            subtitle = shape_text(4, "Subtitle", emu(0.48), emu(0.9), emu(12.2), emu(0.35), self.subtitle, size=13, color="64748B")
        body = "\n".join(self.elements)
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {title}
      {bar}
      {subtitle}
      {body}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""

    def rels_xml(self, media_map):
        rels = []
        for rid, path in self.images:
            rels.append(
                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_map[path]}"/>'
            )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{''.join(rels)}
</Relationships>"""


def write_pptx(slides, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    media_map = {}
    for slide in slides:
        for _, path in slide.images:
            if path not in media_map:
                media_map[path] = f"image{len(media_map) + 1}{path.suffix.lower()}"

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        overrides = [
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Default Extension="png" ContentType="image/png"/>',
            '<Default Extension="jpg" ContentType="image/jpeg"/>',
            '<Default Extension="jpeg" ContentType="image/jpeg"/>',
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
            '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
            '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
        for i in range(1, len(slides) + 1):
            overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        z.writestr("[Content_Types].xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">{"".join(overrides)}</Types>')
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""")
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        z.writestr("docProps/core.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>氦4单相热物性神经网络代理模型组会汇报</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""")
        z.writestr("docProps/app.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex XML PPTX Builder</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{len(slides)}</Slides>
</Properties>""")
        sld_ids = []
        rels = []
        for i in range(1, len(slides) + 1):
            sld_ids.append(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>')
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
        z.writestr("ppt/presentation.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides)+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{''.join(sld_ids)}</p:sldIdLst>
  <p:sldSz cx="{int(SLIDE_W)}" cy="{int(SLIDE_H)}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""")
        z.writestr("ppt/_rels/presentation.xml.rels", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{''.join(rels)}
<Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
</Relationships>""")
        z.writestr("ppt/slideMasters/slideMaster1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>""")
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""")
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""")
        z.writestr("ppt/theme/theme1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office">
  <a:themeElements>
    <a:clrScheme name="Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="16A34A"/></a:accent2><a:accent3><a:srgbClr val="F59E0B"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="DB2777"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>""")
        for i, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide.xml())
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide.rels_xml(media_map))
        for src, media_name in media_map.items():
            z.write(src, f"ppt/media/{media_name}")


def build_slides(root: Path):
    fig = root / "results" / "final_he4_single_phase" / "figures"
    slides = []

    s = Slide("面向低温工程计算的氦4单相热物性神经网络代理模型", "组会汇报 | REFPROP 数据集、多物性 ANN、速度与基线对比")
    s.add_text(0.7, 1.45, 7.2, 1.4, "研究目标：建立一个可替代 REFPROP 高频调用的高精度、快速、可部署物性代理模型", size=24, bold=True)
    s.add_box(0.7, 3.1, 3.2, 1.0, "Fluid\nHelium-4", size=22, fill="DBEAFE", bold=True)
    s.add_box(4.2, 3.1, 3.2, 1.0, "Domain\n5.5-300 K\n0.001-3 MPa", size=20, fill="DCFCE7", bold=True)
    s.add_box(7.7, 3.1, 3.2, 1.0, "Outputs\n8 properties", size=22, fill="FEF3C7", bold=True)
    s.add_text(0.7, 5.7, 9.0, 0.5, "当前状态：模型训练、误差分析、REFPROP 速度对比、传统 ML 基线对比已完成", size=20, color="475569")
    slides.append(s)

    s = Slide("研究背景与问题", "为什么需要 ANN 物性代理模型")
    s.add_text(0.7, 1.35, 5.5, 4.7, "• 低温系统仿真中需要反复调用氦物性\n• REFPROP 精度高，但逐点调用在大规模迭代/优化中较慢\n• 单个坏点可能导致工程迭代发散\n• 目标：在保持 REFPROP 级精度的同时，提高调用速度，并分析误差风险区域", size=22)
    s.add_box(7.0, 1.55, 2.1, 0.85, "T, P", size=26, fill="E0F2FE", bold=True)
    s.add_box(7.0, 2.8, 2.1, 0.85, "REFPROP", size=24, fill="FDE68A", bold=True)
    s.add_box(7.0, 4.05, 2.1, 0.85, "ANN surrogate", size=21, fill="DCFCE7", bold=True)
    s.add_box(9.7, 2.8, 2.4, 2.1, "ρ, s, h, w\nCv, Cp, μ, k", size=22, fill="EEF2FF", bold=True)
    slides.append(s)

    s = Slide("数据集与适用范围", "本文主线：He-4 单相、5.5 K 以上")
    s.add_text(0.7, 1.28, 5.9, 4.9, "• 数据来源：NIST REFPROP 10\n• 相区：单相区\n• 温度范围：5.5-300 K\n• 压力范围：0.001-3 MPa\n• 样本数：296646\n• 输入：Temperature, Pressure\n• 输出：8 个热物性", size=23)
    s.add_box(7.0, 1.35, 4.8, 4.8, "输出物性\n\nDensity\nEntropy\nEnthalpy\nSpeed of sound\nCv / Cp\nViscosity\nThermal conductivity", size=22, fill="F8FAFC")
    slides.append(s)

    s = Slide("模型架构", "多输出 MLP，同时预测 8 个物性")
    s.add_box(0.7, 1.25, 2.0, 0.8, "T, P", size=26, fill="E0F2FE", bold=True)
    s.add_box(3.1, 1.25, 2.5, 0.8, "log10 + scaler", size=22, fill="DBEAFE", bold=True)
    s.add_box(6.0, 1.05, 3.0, 1.2, "MLP\n[256,512,512,256,128]", size=20, fill="DCFCE7", bold=True)
    s.add_box(9.5, 1.25, 2.7, 0.8, "8 properties", size=22, fill="FEF3C7", bold=True)
    s.add_text(0.9, 3.0, 11.2, 3.1, "网络模块：Linear → LayerNorm → GELU → Dropout\n优化器：AdamW，weight decay = 1e-5，cosine annealing\n训练策略：early stopping，多物性联合损失\n优势：结构紧凑，推理速度快，便于工程部署", size=23)
    slides.append(s)

    s = Slide("关键方法：目标变换与加权训练", "解决量纲差异、正值约束和低温困难区域")
    s.add_text(0.7, 1.3, 6.0, 4.8, "目标变换\n• 正值物性：log10(y)\n• 熵：log10(S + 1.0)\n\n加权训练\n• 低温区 5.5-20 K：权重 4.0\n• 过渡区 20-80 K：权重 1.5\n• Density 权重 2.0\n• Entropy 权重 3.0", size=22)
    s.add_text(7.0, 1.4, 5.2, 4.6, "动机\n• 避免密度等正值物性出现非物理解\n• 缓解不同物性尺度差异\n• 强化低温区和困难目标\n• 降低尾部坏点对工程迭代的风险", size=22, color="334155")
    slides.append(s)

    s = Slide("最终模型总体精度", "全测试集统计")
    cards = [
        ("MAPE", "0.150%"),
        ("Median", "0.072%"),
        ("P95", "0.541%"),
        ("P99", "1.126%"),
        ("Mean R²", "0.999951"),
        ("Within 5%", "99.998%"),
    ]
    for i, (k, v) in enumerate(cards):
        x = 0.8 + (i % 3) * 4.1
        y = 1.4 + (i // 3) * 2.0
        s.add_box(x, y, 3.45, 1.25, f"{k}\n{v}", size=25, fill=["DBEAFE", "DCFCE7", "FEF3C7", "EDE9FE", "FFE4E6", "ECFCCB"][i], bold=True)
    s.add_text(0.9, 5.8, 10.8, 0.6, "结论：整体误差低，P99 约 1.13%，绝大多数点落在 5% 以内。", size=22)
    slides.append(s)

    s = Slide("逐物性误差分析", "不同物性的预测难度不同")
    s.add_image(fig / "accuracy" / "property_error_bars.png", 0.6, 1.2, 6.1, 5.5)
    s.add_image(fig / "accuracy" / "property_accuracy_r2.png", 6.8, 1.2, 5.8, 5.5)
    slides.append(s)

    s = Slide("温压平面误差分布与坏点分析", "定位模型风险区域")
    s.add_image(fig / "accuracy" / "error_maps" / "overall_error_heatmap.png", 0.6, 1.2, 5.8, 5.4)
    s.add_image(fig / "accuracy" / "error_maps" / "worst_points_tp_scatter.png", 6.7, 1.2, 5.8, 5.4)
    slides.append(s)

    s = Slide("REFPROP vs ANN 速度对比", "大规模状态点调用")
    s.add_image(fig / "speed" / "refprop_ann_speed.png", 0.6, 1.2, 5.8, 5.4)
    s.add_image(fig / "speed" / "ann_speedup_over_refprop.png", 6.7, 1.2, 5.8, 5.4)
    s.add_text(0.8, 6.75, 11.0, 0.35, "全数据 296646 点：ANN ≈ 418038 points/s，REFPROP ≈ 12113 points/s，加速约 34.5×", size=16, color="475569")
    slides.append(s)

    s = Slide("传统机器学习基线对比", "精度、速度与尾部误差的综合比较")
    s.add_image(fig / "baselines" / "baseline_accuracy_speed.png", 0.6, 1.15, 5.9, 5.45)
    s.add_image(fig / "baselines" / "baseline_tail_errors.png", 6.75, 1.15, 5.7, 5.45)
    slides.append(s)

    s = Slide("基线对比结论", "不能只看平均误差")
    s.add_text(0.75, 1.3, 11.7, 4.8, "• Extra Trees 的平均 MAPE 最低，但推理速度明显低于当前 MLP\n• Random Forest / Gaussian RBF Ridge 平均误差也较低，但最大坏点更大\n• 当前 MLP 的优势：速度最高、最大坏点较小、易部署到工程代码\n• XGBoost 在当前超参数下表现不佳，暂不作为最终方案\n\n组会建议表述：当前模型不是所有单项指标都第一，但在速度、尾部误差和部署便利性上更适合工程计算。", size=23)
    slides.append(s)

    s = Slide("可靠性门控：速度担心如何处理", "门控不等于每个点都调用 REFPROP")
    s.add_text(0.75, 1.3, 6.1, 4.8, "轻量门控只做便宜检查：\n• 输入是否超出训练域\n• 正值物性是否为正\n• Cp 是否大于 Cv\n• 是否落入误差热力图高风险区\n\n只有失败点才 fallback 到 REFPROP。", size=22)
    s.add_box(7.2, 1.5, 4.4, 3.7, "ANN prediction\n↓\ncheap physics/range checks\n↓\npass: use ANN\nfail: call REFPROP", size=24, fill="EEF2FF", bold=True)
    s.add_text(7.1, 5.55, 4.8, 0.7, "只要 fallback 比例很小，整体速度仍接近 ANN。", size=21, color="334155")
    slides.append(s)

    s = Slide("当前工作完成情况", "可作为论文主体结果")
    s.add_text(0.75, 1.25, 11.6, 5.4, "已完成：\n• He4 单相 REFPROP 数据集整理\n• 多输出 ANN 模型训练与优化\n• log / shifted-log target 改进\n• 逐物性指标、R²、RMSE、P95/P99/max 统计\n• 温压平面误差热力图与 worst-point 分析\n• REFPROP 大规模速度对比\n• RF、Extra Trees、XGBoost、高斯核近似基线对比\n• 论文结果包整理", size=23)
    slides.append(s)

    s = Slide("下一步计划", "面向论文与工程应用")
    s.add_text(0.75, 1.3, 11.5, 4.9, "短期：\n• 整理文献对比表\n• 增加物理一致性检查表\n• 进一步优化 MLP 或尝试小型 ensemble\n• 完成可靠性门控的速度影响评估\n\n中期：\n• 构建 5.5 K 以上单相氦4管路/换热器应用算例\n• 写论文初稿：数据、方法、误差、速度、基线对比", size=23)
    slides.append(s)

    return slides


def main():
    root = Path(__file__).resolve().parents[1]
    slides = build_slides(root)
    out = root / "results" / "final_he4_single_phase" / "group_meeting_he4_ann_report.pptx"
    write_pptx(slides, out)
    print(f"Saved PPTX to: {out}")
    shutil.copy2(out, root / "group_meeting_he4_ann_report.pptx")
    print(f"Copied PPTX to: {root / 'group_meeting_he4_ann_report.pptx'}")


if __name__ == "__main__":
    main()
