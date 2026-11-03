#!/bin/bash

# This is a local version of CI testing.
# Unfortunately it has to be kept up to date with ci.yaml and
# ci_examples.yaml (the notebooks section).

# Notes:
# * This is developed only on my Mac M1, so there are implicit assumptions
#   around that which may not work on other platforms or on M1 machines
#   which are not set up identically.

# local configuration
pytest_n=8
# sagehen_gridded: xdist workers x that domain's memory footprint OOMs
# small machines (CI runs -n=1 on Linux, -n=2 elsewhere); keep it low here
pytest_n_gridded=2
# should probably clone mf6 locally and checkout latest develop
# modflow_repo_location=../../modflow6_for_pws_ci

# options
# all "no data" options. if passed, these turn OFF sections of the tests.
while getopts 'hilmtoscrdufgn' opt; do
    case "$opt" in
    h)
        h=h
        echo "Printing HELP:"
        ;;
    i)
        i=i
        echo "Not testing or re-installing pywatershed"
        ;;
    l)
        l=l
        echo "Not linting pywatershed"
        ;;
    m)
        m=m
        echo "Not updating or building Modflow6"
        ;;
    t)
        t=t
        echo "Not running the tests"
        ;;
    o)
        o=o
        echo "Not running the domainless tests"
        ;;
    s)
        s=s
        echo "Not running the sagehen_5yr tests"
        ;;
    c)
        c=c
        echo "Not running the sagehen_gridded_5yr (cascades) tests"
        ;;
    r)
        r=r
        echo "Not running the hru_1 tests"
        ;;
    d)
        d=d
        echo "Not running the drb_2yr tests"
        ;;
    u)
        u=u
        echo "Not running the ucb_2yr tests"
        ;;
    f)
        f=f
        echo "Not running the fgr_ag_2yr tests"
        ;;
    g)
        g=g
        echo "Not populating test data from the oracle for any domain"
        ;;
    n)
        n=n
        echo "Not running the example notebooks"
        ;;
    esac
done
shift "$(($OPTIND - 1))"

if [ ! -z "${h}" ]; then
    echo "Using '-' infront of a letter turns that section off"
    echo
    echo "Structure of options"
    echo "--------------------"
    echo "i: installation"
    echo "l: linting"
    echo "m: modflow update and build"
    echo "n: example notebooks (autotest_exs, as in ci_examples.yaml)"
    echo "t: tests"
    echo "  o: domainless tests"
    echo "  s: sagehen_5yr"
    echo "    g: populate sagehen_5yr from the oracle"
    echo "  c: sagehen_gridded_5yr (cascades)"
    echo "    g: populate sagehen_gridded_5yr from the oracle"
    echo "  r: hru_1"
    echo "    g: populate hru_1 from the oracle"
    echo "  d: drb_2yr"
    echo "    g: populate drb_2yr from the oracle"
    echo "  u: ucb_2yr"
    echo "    g: populate ucb_2yr from the oracle"
    echo "  f: fgr_ag_2yr"
    echo "    g: populate fgr_ag_2yr from the oracle"

    exit 0
fi

echo ""
echo ""

start_dir=$(pwd)

# Populate test_data/<domain> from the oracle answer set: links to its
# inputs and answers, the derived answer variables written alongside.
# The answer set is PWS_ORACLE_DIR (a pws-prms-oracle clone's domains/)
# or, unset, the release assets pinned in prms_oracle.toml, which
# oracle_fetch.py downloads into test_data/.oracle/ first. Arguments
# after the domain are passed on to populate (--control_pattern=...).
populate_from_oracle() {
    domain="$1"
    shift

    echo
    echo ".........."
    echo "$domain - populate test data from the oracle"
    echo ".........."
    echo
    if [ -z "${PWS_ORACLE_DIR}" ]; then
        python ../test_data/generate/oracle_fetch.py \
            --domain="$domain" || return 1
    fi
    python ../test_data/generate/oracle_populate.py \
        --domain="$domain" "$@" || return 1
}

# name: Set environment variables
# export $(head -n1 ../.mf6_ci_ref_remote)
# export $(tail -n1 ../.mf6_ci_ref_remote)

# The mf6 binaries (downloaded by the modflow section, -m to skip)
# persist in ../bin between invocations; always put them on PATH so
# tests needing mf6 (e.g. test_mmr_to_mf6_dfw.py) find them even when
# the modflow section is skipped this invocation.
export PATH="$PATH:$(cd .. && pwd)/bin"

if [ -z "${i}" ]; then
    echo
    echo
    echo "******************************"
    echo "Installation"
    echo "******************************"
    echo

    # run from repository root
    cd ..

    pip uninstall -y pywatershed || exit 1

    ## name: Upgrade pip and install build and twine
    python -m pip install --upgrade pip || exit 1
    pip install wheel build twine importlib_metadata || exit 1
    ## name: Base installation
    pip --verbose install . || exit 1

    ## name: Print pyhmn version
    python -c "import pywatershed; print(pywatershed.__version__)" || exit 1

    ## name: Build pywatershed, check dist outputs
    python -m build || exit 1
    twine check --strict dist/* || exit 1

    cd $start_dir || exit 1
fi

if [ -z "${l}" ]; then
    echo
    echo
    echo "******************************"
    echo "Linting: check and format"
    echo "******************************"
    echo

    # run from repository root
    cd ..
    ruff check . || exit 1
    ruff format --check . || exit 1
    cd $start_dir || exit 1
fi

if [ -z "${m}" ]; then
    echo
    echo
    echo "******************************"
    echo "Modflow6 Update from nighly build"
    echo "******************************"
    echo

    python -m flopy.mf6.utils.generate_classes --ref develop
    get-modflow ../bin --repo modflow6-nightly-build
    # (../bin is already on PATH; exported unconditionally above)

    # The above uses flopy tools to get the nightly build binary and
    # to generate flopy classes against develop. There's a remote possibility
    # those could be out of sync.
    #
    # I'm leaving everything below in case an MF6 local build is desired.

#     # name: Enforce MF6 ref and remote merge to main
#     req_ref=develop # if not develop, submit an issue
#     echo $MF6_REF
#     if [[ "$MF6_REF" != "$req_ref" ]]; then exit 1; fi
#     req_remote=MODFLOW-USGS/modflow6
#     echo $MF6_REMOTE
#     if [[ "$MF6_REMOTE" != "$req_remote" ]]; then
#         echo "bad mf6 remote in .mf6_ci_ref_remote"
#         exit 1
#     fi

#     # Checkout MODFLOW 6 (from $start_dir)
#     if [ ! -d $modflow_repo_location ]; then
#         git clone git@github.com:$req_remote $modflow_repo_location || exit 1
#     fi
#     cd "${modflow_repo_location}" || exit 1
#     git checkout $req_ref || exit 1
#     git fetch origin || exit 1
#     git merge origin/$req_ref || exit 1

#     # Update flopy MODFLOW 6 classes in the current environment
#     cd autotest || exit 1
#     python -m flopy.mf6.utils.generate_classes || exit 8

#     # Build mf6 locally instead of installing mf6 nightly build
#     # install conda env for mf6
#     cd "${modflow_repo_location}" || exit 1
#     env_name=mf64ci
#     # only necessary the first time - create env if it doesn't exist
#     if ! conda env list | grep -q "^${env_name} "; then
#         env_file=environment.yml
#         mamba remove -y --name $env_name --all || exit 1
#         mamba create -y --name $env_name || exit 1
#         mamba env update --name $env_name --file $env_file --prune || exit 1
#     fi

#     conda_dir=$(dirname $CONDA_EXE)
#     source $conda_dir/activate $env_name || exit 1
#     # putting this here b/c of some issues on macos 26
#     # only necessary on macOS
#     if [[ "$OSTYPE" == "darwin"* ]]; then
#         export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
#         export LIBRARY_PATH="$LIBRARY_PATH:$SDKROOT/usr/lib"
#     fi
#     if [ ! -d "buildir" ]; then
#         meson setup --prefix=$(pwd) --libdir=bin builddir || exit 11
#     fi
#     meson install -C builddir || exit 12
#     conda deactivate

#     cd $start_dir
#     export PATH=$PATH:$modflow_repo_location/bin

fi

if [ -z "${t}" ]; then
    echo
    echo
    echo "******************************"
    echo "TESTS"
    echo "******************************"
    echo

    cd ..

    echo
    echo "Get GIS files for tests"
    python -m pywatershed.utils.gis_files || exit 1

    echo
    echo "Get additional domain files for tests"
    python -m pywatershed.utils.addtl_domain_files || exit 1

    cd autotest

    if [ -z "${o}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAINLESS"
        echo "===================="
        echo
        echo "domainless - run tests not requiring domain data"
        pytest -m domainless -n=$pytest_n -vv --error-for-skips || exit 1
    fi

    if [ -z "${s}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAIN: sagehen_5yr"
        echo "===================="
        echo

        if [ -z "${g}" ]; then
            populate_from_oracle sagehen_5yr \
                --control_pattern=sagehen_no_cascades.control || exit 1
        fi

        # - name: sagehen_5yr_no_cascades - list netcdf input files
        #   working-directory: test_data
        #   run: |
        #     find sagehen_5yr/output_no_cascades -name '*.nc'

        echo
        echo ".........."
        echo "sagehen_5yr_no_cascades - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=sagehen_5yr \
            --control_pattern=sagehen_no_cascades.control \
            --durations=0 \
            --error-for-skips \
            --ignore=test_data_model.py \
            --ignore=test_cbh_to_netcdf.py \
            --ignore=test_prms_dyn_params.py \
            --ignore=test_control_read.py \
            --ignore=test_domain_subset.py \
            --ignore=test_mmr_to_mf6_dfw.py \
            --ignore=test_model.py \
            --ignore=test_netcdf_subset.py \
            --ignore=test_obsin_flow_node.py \
            --ignore=test_output.py \
            --ignore=test_pass_through_flow_graph.py \
            --ignore=test_preprocess_cascades.py \
            --ignore=test_prms_param_separate_cascades.py \
            --ignore=test_prms_atmosphere_transp_frost.py \
            --ignore=test_prms_atmosphere_transp_frost_dynamic.py \
            --ignore=test_prms_channel.py \
            --ignore=test_prms_channel_flow_graph.py \
            --ignore=test_prms_et.py \
            --ignore=test_prms_et_can_runoff.py \
            --ignore=test_prms_et_canopy.py \
            --ignore=test_prms_hydraulic_geometry.py \
            --ignore=test_prms_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag_restart.py \
            --ignore=test_prms_runoff.py \
            --ignore=test_prms_runoff_ag.py \
            --ignore=test_prms_runoff_ag_restart.py \
            --ignore=test_prms_runoff_soilzone_ag.py \
            --ignore=test_prms_stream_temp.py \
            --ignore=test_source_sink_flow_node.py \
            --ignore=test_starfit_flow_graph.py || exit 1

        if [ -z "${g}" ]; then
            populate_from_oracle sagehen_5yr \
                --control_pattern=sagehen_no_gw_cascades.control || exit 1
        fi

        echo
        echo ".........."
        echo "sagehen_5yr_no_gw_cascades - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=sagehen_5yr \
            --control_pattern=sagehen_no_gw_cascades.control \
            --durations=0 \
            --error-for-skips \
            test_preprocess_cascades.py \
            test_prms_param_separate_cascades.py \
            test_prms_above_snow.py \
            test_prms_atmosphere.py \
            test_prms_below_snow.py \
            test_prms_canopy.py \
            test_prms_groundwater.py \
            test_restart_processes.py \
            test_prms_snow.py \
            test_prms_solar_geom.py \
            test_self_drive.py || exit 1
    fi

    if [ -z "${c}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAIN: sagehen_gridded_5yr"
        echo "===================="
        echo
        if [ -z "${g}" ]; then
            # the oracle holds the CBH forcing files its make_cbh_only
            # control produced; they are linked with the other inputs
            populate_from_oracle sagehen_gridded_5yr \
                --control_pattern=sagehen_gridded_cascades.control || exit 1
        fi

        echo
        echo ".........."
        echo "sagehen_gridded_5yr - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n_gridded \
            -m "not domainless" \
            --domain=sagehen_gridded_5yr \
            --durations=0 \
            --error-for-skips \
            test_prms_param_separate_cascades.py \
            test_prms_above_snow.py \
            test_prms_below_snow.py \
            test_prms_groundwater.py \
            test_restart_processes.py \
            test_prms_snow.py \
            test_prms_solar_geom.py || exit 1
    fi

    if [ -z "${r}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAIN: hru_1"
        echo "===================="
        echo

        if [ -z "${g}" ]; then
            populate_from_oracle hru_1 \
                --control_pattern=nhm.control \
                --control_pattern=frost.control || exit 1
        fi

        echo
        echo ".........."
        echo "hru_1_nhm - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=hru_1 \
            --control_pattern=nhm.control \
            --durations=0 \
            --error-for-skips \
            --ignore=test_data_model.py \
            --ignore=test_domain_subset.py \
            --ignore=test_prms_dyn_params.py \
            --ignore=test_mmr_to_mf6_dfw.py \
            --ignore=test_obsin_flow_node.py \
            --ignore=test_output.py \
            --ignore=test_pass_through_flow_graph.py \
            --ignore=test_preprocess_cascades.py \
            --ignore=test_prms_param_separate_cascades.py \
            --ignore=test_prms_atmosphere_transp_frost.py \
            --ignore=test_prms_atmosphere_transp_frost_dynamic.py \
            --ignore=test_prms_channel_flow_graph.py \
            --ignore=test_prms_hydraulic_geometry.py \
            --ignore=test_prms_runoff_ag.py \
            --ignore=test_prms_runoff_ag_restart.py \
            --ignore=test_prms_runoff_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag_restart.py \
            --ignore=test_prms_stream_temp.py \
            --ignore=test_source_sink_flow_node.py \
            --ignore=test_starfit_flow_graph.py || exit 1

        echo
        echo ".........."
        echo "hru_1_nhm_transp_frost - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=hru_1 \
            --control_pattern=nhm_transp_frost.control \
            --durations=0 \
            --error-for-skips \
            test_prms_atmosphere_transp_frost.py || exit 1

    fi

    if [ -z "${d}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAIN: drb_2yr"
        echo "===================="
        echo

        if [ -z "${g}" ]; then
            populate_from_oracle drb_2yr || exit 1
        fi

        echo ".........."
        echo "drb_2yr_nhm - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=drb_2yr \
            --control_pattern=nhm.control \
            --durations=0 \
            --error-for-skips \
            --ignore=test_obsin_flow_node.py \
            --ignore=test_preprocess_cascades.py \
            --ignore=test_prms_param_separate_cascades.py \
            --ignore=test_prms_dyn_params.py \
            --ignore=test_prms_atmosphere_transp_frost.py \
            --ignore=test_prms_atmosphere_transp_frost_dynamic.py \
            --ignore=test_prms_hydraulic_geometry.py \
            --ignore=test_prms_stream_temp.py \
            --ignore=test_domain_subset.py \
            --ignore=test_prms_runoff_ag.py \
            --ignore=test_prms_runoff_ag_restart.py \
            --ignore=test_prms_runoff_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag_restart.py \
            --ignore=test_source_sink_flow_node.py \
            --ignore=test_starfit_flow_graph.py || exit 1

        # Specific tests not redundant with dprst
        echo ".........."
        echo "drb_2yr_no_dprst - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=drb_2yr \
            --control_pattern=no_dprst \
            --durations=0 \
            --error-for-skips \
            test_prms_runoff.py \
            test_prms_soilzone.py \
            test_prms_groundwater.py \
            test_prms_above_snow.py \
            test_prms_below_snow.py \
            test_restart_processes.py || exit 1

        # # Specific tests not redundant with dprst
        echo ".........."
        echo "drb_2yr_obsin - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=0 \
            -m "not domainless" \
            --domain=drb_2yr \
            --control_pattern=nhm_obsin.control \
            --durations=0 \
            --error-for-skips \
            test_obsin_flow_node.py || exit 1

        echo ".........."
        echo "drb_2yr_transp_frost - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            --domain=drb_2yr \
            --control_pattern=nhm_transp_frost.control \
            --durations=0 \
            --error-for-skips \
            test_prms_atmosphere_transp_frost.py || exit 1

        echo ".........."
        echo "drb_2yr_nhm_stream_temp - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            --domain=drb_2yr \
            --control_pattern=nhm_stream_temp \
            --durations=0 \
            --error-for-skips \
            test_prms_hydraulic_geometry.py \
            test_prms_stream_temp.py || exit 1

    fi

    if [ -z "${u}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAIN: ucb_2yr"
        echo "===================="
        echo

        if [ -z "${g}" ]; then
            populate_from_oracle ucb_2yr \
                --control_pattern=nhm.control \
                --control_pattern=frost.control || exit 1
        fi

        echo ".........."
        echo "ucb_2yr_nhm - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=ucb_2yr \
            --control_pattern=nhm.control \
            --durations=0 \
            --error-for-skips \
            --ignore=test_data_model.py \
            --ignore=test_netcdf_subset.py \
            --ignore=test_prms_dyn_params.py \
            --ignore=test_obsin_flow_node.py \
            --ignore=test_output.py \
            --ignore=test_pass_through_flow_graph.py \
            --ignore=test_preprocess_cascades.py \
            --ignore=test_prms_param_separate_cascades.py \
            --ignore=test_prms_atmosphere_transp_frost.py \
            --ignore=test_prms_atmosphere_transp_frost_dynamic.py \
            --ignore=test_mmr_to_mf6_dfw.py \
            --ignore=test_prms_hydraulic_geometry.py \
            --ignore=test_prms_stream_temp.py \
            --ignore=test_prms_runoff_ag.py \
            --ignore=test_prms_runoff_ag_restart.py \
            --ignore=test_prms_runoff_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag.py \
            --ignore=test_prms_soilzone_ag_restart.py || exit 1

        echo ".........."
        echo "ucb_2yr_nhm_transp_frost - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            -m "not domainless" \
            --domain=ucb_2yr \
            --control_pattern=nhm_transp_frost.control \
            --durations=0 \
            --error-for-skips \
            test_prms_atmosphere_transp_frost.py || exit 1
    fi

    if [ -z "${f}" ]; then
        echo
        echo
        echo "===================="
        echo "DOMAIN: fgr_ag_2yr"
        echo "===================="
        echo

        if [ -z "${g}" ]; then
            populate_from_oracle fgr_ag_2yr \
                --control_pattern=spinup.control \
                --control_pattern=analysis.control || exit 1
        fi

        echo ".........."
        echo "fgr_ag_2yr - pywatershed tests"
        echo ".........."
        echo
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            --domain=fgr_ag_2yr \
            --control_pattern=spinup.control \
            --control_pattern=analysis.control \
            --durations=0 \
            --error-for-skips \
            test_prms_runoff_ag.py \
            test_prms_runoff_ag_restart.py \
            test_prms_soilzone_ag.py \
            test_prms_soilzone_ag_restart.py \
            test_prms_runoff_soilzone_ag.py \
            test_prms_dyn_params.py || exit 1

        echo ".........."
        echo "fgr_ag_2yr_analysis transp_frost_dynamic - pywatershed tests"
        echo ".........."
        echo
        # analysis.control only: these tests skip without dynamic frost dates
        pytest \
            -vv \
            -rs \
            -n=$pytest_n \
            --domain=fgr_ag_2yr \
            --control_pattern=analysis.control \
            --durations=0 \
            --error-for-skips \
            test_prms_atmosphere_transp_frost_dynamic.py || exit 1

    fi
fi

if [ -z "${n}" ]; then
    echo
    echo
    echo "******************************"
    echo "EXAMPLE NOTEBOOKS (ci_examples.yaml)"
    echo "******************************"
    echo

    # the domains the notebooks read (03/04/08/09: drb_2yr, 11: sagehen_5yr)
    populate_from_oracle drb_2yr || exit 1
    populate_from_oracle sagehen_5yr || exit 1

    cd ..
    echo "Get GIS files for notebooks"
    python -m pywatershed.utils.gis_files || exit 1
    echo "Get additional domain files for notebooks"
    python -m pywatershed.utils.addtl_domain_files || exit 1

    # Notebook 07 needs the mf6 binary from the modflow section (../bin,
    # already on PATH).
    cd autotest_exs || exit 1
    pytest -s -vv --durations=0 || exit 1
    cd $start_dir || exit 1
fi

if [ -z "${i}" ]; then
    cd $start_dir || exit 3
    # If install was done, put the install back to its original, editable state.
    # Do it here so the tests use the test install if it is done.
    pip uninstall -y pywatershed || exit 1
    cd .. || exit 1
    pip install -e . || exit 2

    cd $start_dir || exit 3
fi

exit 0
