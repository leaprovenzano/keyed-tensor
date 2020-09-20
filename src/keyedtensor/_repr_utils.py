from typing import List


def format_field(kstr: str, rows: List[str], prefix: str = '') -> str:
    kprefix = f'{prefix}{kstr}'
    pad = ' ' * (len(kprefix))
    tensorstr = '\n'.join(f'{pad if i > 0 else ""}{v}' for i, v in enumerate(rows))
    return f'{kprefix}{tensorstr}'


def keyedtensor_str(kt) -> str:
    prefix = f'{kt.__class__.__name__}('
    str_data = [(f'{k}=', f'{t!r}'.split('\n')) for k, t in kt.items()]
    if all(len(t) == 1 for _, t in str_data):
        return f'{prefix}{", ".join(f"{k}{v[0]}" for k, v in str_data)})'
    return (
        ',\n'.join(
            format_field(kstr, rows, prefix=' ' * len(prefix) if i else prefix)
            for i, (kstr, rows) in enumerate(str_data)
        )
        + ')'
    )
