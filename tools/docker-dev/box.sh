#! /usr/bin/env bash

# Stops the execution of this script if an error occurs.
set -e

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
NC="\033[0m"

DEFAULT_DOCKER_IMAGE="horizonrobotics/alf:0.0.6-pytorch1.8-python3.7"

function box::cli_help() {
    echo -e "Usage: ${BLUE}./box.sh${NC} [DOCKER_IMAGE]"
    echo ""
    echo "  Start an ephemeral standard docker container as Alf development environment"
    echo ""
    echo "Options:"
    echo "  DOCKER_IMAGE: specify the docker image, or otherwise the default will be used."
}

# ---- Logging Helpers ----

function box::ok() {
    echo -e "[ ${GREEN}ok${NC} ] $1"
}

function box::fail() {
    echo -e "[${RED}FAIL${NC}] $1"
    exit -1
}

function box::warn() {
    echo -e "[${YELLOW}WARN${NC}] $1"
}

function box::info() {
    echo -e "[${BLUE}info${NC}] $1"
}

# ---- Actual Implementation ----

function box::get_script_dir() {
    echo "$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
}

function box::get_alf_dir() {
    local script_dir="$(box::get_script_dir)"
    local tools_dir="$(dirname ${script_dir})"
    local alf_dir="$(dirname ${tools_dir})"
    echo "${alf_dir}"
}

function box::init_container() {
    local alf_dir=$1
    local script_dir=$2
    local image=$3

    # Make sure that docker is installed.
    if ! [ -x "$(command -v docker)" ]; then
        box::fail "Command 'docker' not found. Please install docker first."
    fi

    # This tests whether ${alf_dir} is under the user's home directory. This is
    # an assumption through out this script.
    if [[ "${alf_dir##${HOME}}" = "${alf_dir}" ]]; then
        box::fail "Alf directory ${alf_dir} is not under your home directory ${HOME}"
    else
        box::ok "Confirmed that alf directory is ${alf_dir}, which is under ${HOME}"
    fi

    local container=$(docker ps -a -q -f name=alf-dev-box)

    if [ ! -z "${container}" ]; then
        local exited=$(docker ps -aq -f status=exited -f name=alf-dev-box)
        if [ ! -z "${exited}" ]; then
            # The existing container alf-dev-box has exited, so we can safely remove it.
            docker rm "${exited}"
            box::ok "Deprecated the already exited alf-dev-box"
        else
            box::info "There is an active container named alf-dev-box, and it will be used."
            return 0
        fi
    fi

    box::info "Launching docker container from ${image} ..."

    docker run -dit \
           --name "alf-dev-box" \
           --user $UID:$GID \
           -v "/etc/passwd:/etc/passwd:ro" \
           -v "/etc/group:/etc/group:ro" \
           -v "/etc/shadow:/etc/shadow:ro" \
           -v "/home/${USER}:/home/${USER}" \
           -v "${script_dir}/bashrc.override:/home/${USER}/.bashrc:ro" \
           -v "${script_dir}/inputrc.override:/home/${USER}/.inputrc:ro" \
           -v "/tmp.X11-unix:/tmp/.x11-unix:ro" \
           --workdir "${alf_dir}" \
           --network host \
           ${image} /bin/bash

    container=$(docker ps -a -q -f name=alf-dev-box)

    docker exec -u 0 alf-dev-box /bin/bash -c "apt update && apt install -y rsync"
    docker exec alf-dev-box /bin/bash -c "pip3 install -e ${alf_dir}"
    box::ok "Successfully launched alf-dev-box with id ${container}"
}

function box::enter_container() {
    box::ok "Entering the container"
    docker exec -it alf-dev-box /bin/bash
}

function box::main() {
    local argument="${DEFAULT_DOCKER_IMAGE}"

    if [ "$#" -eq 1 ]; then
        argument="$1"
    fi

    case "$argument" in
        --help)
            box::cli_help
            ;;
        *)
            local alf_dir="$(box::get_alf_dir)"
            local script_dir="$(box::get_script_dir)"

            box::init_container "${alf_dir}" "${script_dir}" "${argument}"

            local success=$?
            if [ ! "${success}" -eq "0" ]; then
                box:fail "Running into error in launching the container."
            fi

            box::enter_container

            # At this point the user finished operating in the container.
            box::ok "Leaving the alf-dev-box container. The container remains active."
            ;;
    esac
}

box::main $@
