from mccode_antlr.loader.loader import parse_mcstas_instr
from textwrap import dedent
from pytest import mark
from functools import cache

def compiled(method):
    from mccode_antlr.compiler.check import simple_instr_compiles
    if simple_instr_compiles('cc'):
        return method
    @mark.skip(reason=f"Working C compiler required for {method}")
    def skipped_method(*args, **kwargs):
        return method(*args, **kwargs)
    return skipped_method


def compile_and_run(instr, parameters, run=True, dump=False):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from mccode_antlr.translators.target import MCSTAS_GENERATOR
    from mccode_antlr.run import mccode_compile, mccode_run_compiled

    kwargs = dict(generator=MCSTAS_GENERATOR, dump_source=dump)
    with TemporaryDirectory() as directory:
        binary, target = mccode_compile(instr, directory, **kwargs)
        # pick a not-yet-created folder for instrument output
        out = Path(directory).joinpath('t')
        result, dat_files = None, None
        if run:
            result, dat_files = mccode_run_compiled(binary, target, out, parameters)
        return result, dat_files
        

def this_registry():
    from git import Repo
    from pathlib import Path
    from mccode_antlr.reader.registry import LocalRegistry
    try:
        repo = Repo('.', search_parent_directories=True) 
        root = repo.working_tree_dir
        return LocalRegistry('this_registry', root)
    except ex:
        raise RuntimeError(f"Unable to identify base repository, {ex}")


def detector_tubes():
    instr = parse_mcstas_instr(dedent("""
    DEFINE INSTRUMENT py_test_detector_tubes(int which=0)
    DECLARE %{
    double diameter=0.1;
    double raster_height;
    int raster_count=100;
    %}
    USERVARS %{
    int charge_left;
    int charge_right;
    %}
    INITIALIZE %{
    raster_height = 1.0 / (double)raster_count;
    %}
    TRACE
    COMPONENT source = Arm() AT (0, 0, 0) ABSOLUTE
    EXTEND %{
      long long n = mcget_run_num();
      long long i = n % 5;
      long long j = (n / 5) % raster_count;
      x = ((double)i - 2.0) * diameter; // odd number of pixels/tubes
      y = ((double)j - raster_count/2 + 0.5) * raster_height; // even number of pixels
      z = 0;
      vx = 0;
      vy = 0;
      vz = 1;
      p = 1.0;
    %}

    COMPONENT ten_tubes = Detector_tubes(
      width=5*diameter+1e-6, height=raster_count * raster_height, radius=diameter/2, wires_in_series=1, no=10, N=5,
      rho=4000.0, R=2000.0, dead_length=0, wire_filename="wire", pack_filename="pack",
      charge_a="charge_left", charge_b="charge_right"
    ) AT (0, 0, 1) ABSOLUTE
    EXTEND %{
    if (SCATTERED){
      // Abuse the position to calcualte (and record) the charge division:
      x = (double)charge_left / (double)(charge_right + charge_left);
      y = 0;
    }
    %}

    COMPONENT recorder = PSDlin_monitor(
      xmin=0, xmax=1, ymin=-0.1, ymax=0.1, nbins=14, filename="division"
    ) AT (0, 0, 2) ABSOLUTE

    FINALLY %{
    %}
    END
    """), registries=[this_registry()])
    results, files = compile_and_run(instr, '-n 1000 which=0', dump=False)
    lines = results.decode('utf-8').splitlines()
    return lines, files


@compiled
def test_detector_tubes():
    from numpy import sum, std, abs
    lines, files = detector_tubes()

    assert 'wire' in files
    assert 'pack' in files
    assert 'division' in files
    # verify that the output files are as expected ...
    
    pack = files['pack'].structured['I']
    assert sum(pack[:]) == 1000, "Every produced ray should be detected"
    assert std(pack[:]) == 0, "All 2-D pixels should be hit the same number of times"
    
    wire = files['wire'].structured['I']
    assert sum(wire) == 1000, "The wire output indexes the same pixelated space"
    assert std(wire) == 0
    
    # Now the real test for charge division correctness. The ratios of (1m)*rho and R have been chosen as 2:1.
    # This means that each tube should be twice as long as a gap in charge-division; so the whole
    # space needs to be divisible by 14 = 2*5 + 4 to have an integer number of bins per section.
    division = files['division'].structured['I']
    gaps = division[[2, 5, 8, 11]]
    assert sum(gaps) < 5, "Raster/randomness might put one event per gap"
    assert abs(sum(division)-sum(wire)) < 10, "All events should show up in the division signal, but one might be missing"
    


