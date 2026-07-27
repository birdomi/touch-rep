#!/usr/bin/env python3
"""Generate four printable large ArUco markers (DICT_4X4_50, IDs 0-3)."""

import argparse
from pathlib import Path

import cv2
from PIL import Image


MARKER_IDS = range(4)
MODULES = 6
PIXELS = 1200


def get_marker(dictionary, marker_id):
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, PIXELS, borderBits=1)
    step = PIXELS // MODULES
    cells = [
        [int(marker[row * step + step // 2, col * step + step // 2]) for col in range(MODULES)]
        for row in range(MODULES)
    ]
    return marker, cells


def make_svg(cells, size_mm):
    rectangles = []
    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            if value == 0:
                rectangles.append(f'  <rect x="{col}" y="{row}" width="1" height="1"/>')
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_mm:g}mm" '
            f'height="{size_mm:g}mm" '
            'viewBox="0 0 6 6" shape-rendering="crispEdges">',
            '  <rect width="6" height="6" fill="white"/>',
            '  <g fill="black">',
            *rectangles,
            '  </g>',
            '</svg>',
            '',
        ]
    )


def pdf_object(number, body):
    return f"{number} 0 obj\n{body}\nendobj\n".encode("ascii")


def page_stream(cells, marker_id, size_mm):
    pt_per_mm = 72 / 25.4
    page_w = 210 * pt_per_mm
    page_h = 297 * pt_per_mm
    marker = size_mm * pt_per_mm
    module = marker / MODULES
    x = ((210 - size_mm) / 2) * pt_per_mm
    y = page_h - (70 + size_mm) * pt_per_mm

    commands = ["1 1 1 rg 0 0 %.4f %.4f re f" % (page_w, page_h), "0 0 0 rg"]
    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            if value == 0:
                rx = x + col * module
                ry = y + (MODULES - 1 - row) * module
                commands.append(f"{rx:.4f} {ry:.4f} {module:.4f} {module:.4f} re f")

    gap = 2 * pt_per_mm
    length = 5 * pt_per_mm
    commands.extend(
        [
            "0.35 G 0.35 w",
            f"{x-gap-length:.4f} {y:.4f} m {x-gap:.4f} {y:.4f} l S",
            f"{x:.4f} {y-gap-length:.4f} m {x:.4f} {y-gap:.4f} l S",
            f"{x+marker+gap:.4f} {y:.4f} m {x+marker+gap+length:.4f} {y:.4f} l S",
            f"{x+marker:.4f} {y-gap-length:.4f} m {x+marker:.4f} {y-gap:.4f} l S",
            f"{x-gap-length:.4f} {y+marker:.4f} m {x-gap:.4f} {y+marker:.4f} l S",
            f"{x:.4f} {y+marker+gap:.4f} m {x:.4f} {y+marker+gap+length:.4f} l S",
            f"{x+marker+gap:.4f} {y+marker:.4f} m {x+marker+gap+length:.4f} {y+marker:.4f} l S",
            f"{x+marker:.4f} {y+marker+gap:.4f} m {x+marker:.4f} {y+marker+gap+length:.4f} l S",
            f"BT /F1 12 Tf 0 g {x:.4f} {y-12*pt_per_mm:.4f} Td "
            f"(DICT_4X4_50  ID {marker_id}  |  {size_mm:g} mm) Tj ET",
            f"BT /F1 9 Tf 0 g {45*pt_per_mm:.4f} {15*pt_per_mm:.4f} Td "
            "(Print at Actual size / 100% scale - do not Fit to page) Tj ET",
        ]
    )
    return ("\n".join(commands) + "\n").encode("ascii")


def make_multipage_pdf(all_cells, size_mm, out_dir):
    pt_per_mm = 72 / 25.4
    page_w = 210 * pt_per_mm
    page_h = 297 * pt_per_mm
    font_object = 11
    page_numbers = [3, 5, 7, 9]
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects = [
        pdf_object(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        pdf_object(2, f"<< /Type /Pages /Kids [{kids}] /Count 4 >>"),
    ]

    for index, cells in enumerate(all_cells):
        page_number = page_numbers[index]
        content_number = page_number + 1
        stream = page_stream(cells, index, size_mm)
        objects.append(
            pdf_object(
                page_number,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
                f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
                f"/Contents {content_number} 0 R >>",
            )
        )
        objects.append(
            pdf_object(
                content_number,
                f"<< /Length {len(stream)} >>\nstream\n{stream.decode('ascii')}endstream",
            )
        )

    objects.append(pdf_object(font_object, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objects.sort(key=lambda item: int(item.split(b" ", 1)[0]))

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
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
    size_label = f"{size_mm:g}"
    (out_dir / f"aruco_4x4_50_ids_0-3_{size_label}mm_A4_4pages.pdf").write_bytes(pdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size-mm",
        type=float,
        default=100,
        help="Black marker edge length in millimetres (default: 100)",
    )
    args = parser.parse_args()
    if args.size_mm <= 0 or args.size_mm > 180:
        parser.error("--size-mm must be greater than 0 and no larger than 180")

    size_label = f"{args.size_mm:g}"
    out_dir = Path(__file__).resolve().parents[1] / f"aruco_markers_{size_label}mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dpi = PIXELS / args.size_mm * 25.4
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    all_cells = []
    for marker_id in MARKER_IDS:
        marker, cells = get_marker(dictionary, marker_id)
        all_cells.append(cells)
        stem = f"aruco_4x4_50_id_{marker_id:02d}_{size_label}mm"
        Image.fromarray(marker).save(out_dir / f"{stem}.png", dpi=(png_dpi, png_dpi))
        (out_dir / f"{stem}.svg").write_text(
            make_svg(cells, args.size_mm), encoding="utf-8"
        )
    make_multipage_pdf(all_cells, args.size_mm, out_dir)


if __name__ == "__main__":
    main()
