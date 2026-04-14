"""
Convert SVG to PNG using CairoSVG. Usage:
  pip install cairosvg
  python svg_to_png.py input.svg output.png --dpi 300

If you prefer Inkscape:
  inkscape input.svg --export-type=png --export-filename=output.png --export-dpi=300
"""
import sys
import os
import argparse

try:
    import cairosvg
except Exception:
    cairosvg = None


def convert(input_svg: str, output_png: str, dpi: int = 300):
    if cairosvg is None:
        raise RuntimeError("cairosvg not installed. Run: pip install cairosvg")
    cairosvg.svg2png(url=input_svg, write_to=output_png, dpi=dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_svg')
    parser.add_argument('output_png')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    if not os.path.exists(args.input_svg):
        print(f"Input file not found: {args.input_svg}")
        sys.exit(2)
    try:
        convert(args.input_svg, args.output_png, dpi=args.dpi)
        print(f"Wrote {args.output_png} (dpi={args.dpi})")
    except Exception as e:
        print("Conversion failed:", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
