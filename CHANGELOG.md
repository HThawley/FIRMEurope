# Changelog

## Unreleased

- Refactors of FIRM_CE codebase
- Adds basic testing 
- Adds CI/CD tools via github actions:
    - black
    - flake8
    - bandit
    - coverage
    - pytest
- Many precharging bugfixes associated with:
    - incorrect +/- signs
    - incorrect adjustments of maximum dispatch arrays for nodes
    - constraints incorrectly based on trickling_reserves instead of remaining_trickling_reserves
    - node.netload_t failing to be updated when resetting transmission for an interval
    - flexible generation not being reset prior to "precharging_adjust_storage" transmission_case
- Added `examples` folder
- Added docstrings
- Replaced TOLERANCE magic numbers with constant
- Fixed transmission bug when multiple routes on same leg end in the same node
- Fixed bug where minor_linor.new_build was not updated when Generator and Storage capacity was built
- Fixed bug in calculation of line build costs where line.length was always assumed to be 0 km, resulting in a cost of $0
- Added logging_flag argument to Model to allow logging to be disabled (log stored in `results/temp` when logging_flag = False)
- Added additional rows of data to capacities.csv result file
- Fixed LCOG to use total_generation as denominator instead of total_energy (demand). Fixed LCOB_spillage_losses calculation too.
- Added model_name as an attribute to Model instance
- Moved data and config folders to `inputs/config` and `inputs/data`
- Uncommented the debug lines for the broad optimum module
- NUM_THREADS can now be set as an environment variable by the user and defaults to os.cpu_count()
- Added functionality to provide CSV input for initial population of each scenario
- Built initial simple result viewer.
- Removed some duplicate code behaviour.

## Releases

### v0.0.1 (2025-10-??)

- TBD