from pathlib import Path

def file_hash(filepath: Path) -> str:
    # Simplified from Pooch -- we only care about sha256 for now
    import hashlib
    import functools
    chunksize = 65536
    hasher = getattr(hashlib, 'sha256', functools.partial(hashlib.new, 'sha256'))()
    with filepath.open('rb') as filein:
        chunk = filein.read(chunksize)
        while chunk:
            hasher.update(chunk)
            chunk = filein.read(chunksize)
    return hasher.hexdigest()


def make_registry(base, dirs, recursive=True, ext=None):
    pat = '**/*' if recursive else '*'
    if ext is not None:
        pat = f'{pat}{ext}'
    hashes = {p.relative_to(base).as_posix(): file_hash(p) for d in dirs for p in base.joinpath(d).glob(pat) if p.is_file()}
    return hashes
    

def write_registry(hashes: dict[str, str], output: Path):
    lines = '\n'.join(f'{name} {hashes[name]}' for name in sorted(hashes.keys()))
    with output.open('w') as file:
        file.write(lines + '\n')


def find_git_root() -> Path:
    import git
    repo = git.Repo(Path(__file__), search_parent_directories=True)
    return Path(repo.git_dir).parent


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, default=find_git_root(), help='register relative to root path')
    parser.add_argument('-d', '--dirs', type=str, default='', help='comma separated list of directories relative to root to include')
    parser.add_argument('-r', '--recursive', action='store_true', help='recurse through list of directories')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='disable recursion')
    parser.add_argument('--ext', type=str, nargs='*', action='append', default=None, help='limit search to only this extension')
    parser.add_argument('-o', '--output', type=str, default='pooch-registry.txt', help='registry file name')
    args = parser.parse_args()

    root = Path(args.root) if isinstance(args.root, str) else args.root
    dirs = [''] if args.dirs == '' else args.dirs.split(',')
    if args.ext:
        hashes = {}
        for ext in args.ext:
            hashes.update(make_registry(root, dirs, recursive=args.recursive, ext=ext))
    else:
        hashes = make_registry(root, dirs, recursive=args.recursive)
    
    if len(hashes):
        write_registry(hashes, root.joinpath(args.output))


if __name__ == '__main__':
    main()
