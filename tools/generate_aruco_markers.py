#!/usr/bin/env python3
"""Generate six printable 30 mm ArUco markers (DICT_4X4_50, IDs 0-5)."""

from pathlib import Path

import cv2
from PIL import Image


OUT_DIR = Path(__file__).resolve().parents[1] / "aruco_markers_30mm"
DICTIONARY_NAME = "DICT_4X4_50"
MARKER_IDS = range(6)
MODULES = 6  # 4 data modules + a one-module black border on every side
PX_PER_MODULE = 200
PNG_DPI = 1016  # 40 px/mm: 1200 px is exactly 30 mm in print metadata


def marker_cells(dictionary, marker_id):
    marker = cv2.aruco.generateImageMarker(
        dictionary, marker_id, MODULES * PX_PER_MODULE, borderBits=1
    )
    return [
        [
            int(marker[row * PX_PER_MODULE + PX_PER_MODULE // 2,
                       col * PX_PER_MODULE + PX_PER_MODULE // 2])
            for col in range(MODULES)
        ]
        for row in range(MODULES)
    ], marker


def svg_for(cells):
    black = []
    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            if value == 0:
                black.append(
                    f'  <rect x="{col * 5}" y="{row * 5}" width="5" height="5"/>'
                )
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="30mm" height="30mm" '
            'viewBox="0 0 30 30" shape-rendering="crispEdges">',
            '  <rect width="30" height="30" fill="white"/>',
            '  <g fill="black">',
            *black,
            '  </g>',
            '</svg>',
            '',
        ]
    )


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_object(number, body):
    return f"{number} 0 obj\n{body}\nendobj\n".encode("ascii")


def make_pdf(all_cells):
    pt_per_mm = 72 / 25.4
    page_w = 210 * pt_per_mm
    page_h = 297 * pt_per_mm
    marker = 30 * pt_per_mm
    module = marker / MODULES

    # Two columns by three rows. Coordinates below are marker top-left corners in mm.
    positions = [(45, 35), (135, 35), (45, 105), (135, 105), (45, 175), (135, 175)]
    commands = ["1 1 1 rg 0 0 %.4f %.4f re f" % (page_w, page_h)]

    for marker_id, (cells, (x_mm, top_mm)) in enumerate(zip(all_cells, positions)):
        x = x_mm * pt_per_mm
        y = page_h - (top_mm + 30) * pt_per_mm
        commands.append("0 0 0 rg")
        for row, values in enumerate(cells):
            for col, value in enumerate(values):
                if value == 0:
                    rx = x + col * module
                    ry = y + (MODULES - 1 - row) * module
                    commands.append(f"{rx:.4f} {ry:.4f} {module:.4f} {module:.4f} re f")

        # Fine crop marks sit outside the marker and are removed when cut.
        gap = 1.5 * pt_per_mm
        length = 4 * pt_per_mm
        commands.append("0.35 G 0.35 w")
        commands.extend(
            [
                f"{x-gap-length:.4f} {y:.4f} m {x-gap:.4f} {y:.4f} l S",
                f"{x:.4f} {y-gap-length:.4f} m {x:.4f} {y-gap:.4f} l S",
                f"{x+marker+gap:.4f} {y:.4f} m {x+marker+gap+length:.4f} {y:.4f} l S",
                f"{x+marker:.4f} {y-gap-length:.4f} m {x+marker:.4f} {y-gap:.4f} l S",
                f"{x-gap-length:.4f} {y+marker:.4f} m {x-gap:.4f} {y+marker:.4f} l S",
                f"{x:.4f} {y+marker+gap:.4f} m {x:.4f} {y+marker+gap+length:.4f} l S",
                f"{x+marker+gap:.4f} {y+marker:.4f} m {x+marker+gap+length:.4f} {y+marker:.4f} l S",
                f"{x+marker:.4f} {y+marker+gap:.4f} m {x+marker:.4f} {y+marker+gap+length:.4f} l S",
            ]
        )
        label = pdf_escape(f"DICT_4X4_50  ID {marker_id}  |  30 mm")
        commands.append(
            f"BT /F1 8 Tf 0 g {x:.4f} {y-10*pt_per_mm:.4f} Td ({label}) Tj ET"
        )

    note = pdf_escape("Print at Actual size / 100% scale (do not Fit to page)")
    commands.append(f"BT /F1 9 Tf 0 g {20*pt_per_mm:.4f} {12*pt_per_mm:.4f} Td ({note}) Tj ET")
    stream = ("\n".join(commands) + "\n").encode("ascii")

    objects = [
        pdf_object(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        pdf_object(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        pdf_object(
            3,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        ),
        pdf_object(4, f"<< /Length {len(stream)} >>\nstream\n" + stream.decode("ascii") + "endstream"),
        pdf_object(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    pdf = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    (OUT_DIR / "aruco_4x4_50_ids_0-5_30mm_A4.pdf").write_bytes(pdf)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    all_cells = []
    for marker_id in MARKER_IDS:
        cells, marker = marker_cells(dictionary, marker_id)
        all_cells.append(cells)
        stem = f"aruco_4x4_50_id_{marker_id:02d}_30mm"
        Image.fromarray(marker).save(
            OUT_DIR / f"{stem}.png", dpi=(PNG_DPI, PNG_DPI), compress_level=9
        )
        (OUT_DIR / f"{stem}.svg").write_text(svg_for(cells), encoding="utf-8")
    make_pdf(all_cells)


if __name__ == "__main__":
    main()
