from cProfile import Profile
from pstats import SortKey, Stats

import os
import sys
import subprocess

import src.figures
import src.simulation

def get_cprofile(func: callable, filename=None, folder="profiling/profiles/", verbose=True):
    with Profile() as profile:
        func()

        if verbose:
            (Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.TIME)
                .print_stats(15))
        if filename is not None:
            profile.dump_stats(os.path.join(
                        folder, filename
                    ))

def _get_git_hash() -> str:
    # This git-hash retrieving helper is by Gemini (at my request)
    #  so I can better track what code corresponds to before and after 
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    get_cprofile(src.figures.main, filename="before__src.figures.main"+_get_git_hash()+".prof")