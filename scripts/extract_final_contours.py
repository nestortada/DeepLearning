"""Extract only official PAPILA v2 contours aligned to the frozen JPEGs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from src.data import _read_manifest
from src.internal_test import _json, sha256_file


OFFICIAL_MD5 = "7d73fc5157d44b221a62ed5a4baf3779"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != OFFICIAL_MD5:
            raise ValueError("El ZIP no coincide con el MD5 publicado en Figshare.")
    records = _read_manifest(args.manifest)
    contents = {}
    with ZipFile(args.archive) as archive:
        images, contours = defaultdict(list), defaultdict(list)
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if "FundusImages" in path.parts and path.suffix.lower() == ".jpg":
                images[path.name].append(member)
            if "Contours" in path.parts and path.suffix == ".txt":
                contours[path.name].append(member)
        for split in ("val", "test"):
            for row in records[split]:
                jpeg = images[row.relative_path.name]
                if len(jpeg) != 1 or hashlib.sha256(archive.read(jpeg[0])).hexdigest() != sha256_file(
                    args.dataset_dir / row.relative_path,
                ):
                    raise ValueError(f"El JPEG local de {row.image_id} difiere de la versión oficial.")
                for suffix in ("disc_exp1", "disc_exp2", "cup_exp1", "cup_exp2"):
                    name = f"{row.image_id}_{suffix}.txt"
                    values = {archive.read(member) for member in contours[name]}
                    if len(values) != 1:
                        raise ValueError(f"Contorno ausente o contradictorio: {name}")
                    contents[name] = values.pop()
    args.output.mkdir(parents=True)
    for name, content in contents.items():
        (args.output / name).write_bytes(content)
    _json(args.output / "source.json", {
        "figshare_file_id": 35013982, "archive_md5": OFFICIAL_MD5, "contours": len(contents),
        "jpeg_matches": len(records["val"]) + len(records["test"]),
        "manifest_sha256": sha256_file(args.manifest),
    })


if __name__ == "__main__":
    main()
