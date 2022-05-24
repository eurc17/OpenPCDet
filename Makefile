.PHONY: build clean build_torch clean_all list

ENPTY_TARGETS = .install_torch .install_btcdet .install_spconv .build_poetry

default: build

build: .install_torch .install_btcdet .install_spconv .build_poetry

build_torch: .install_torch 

.build_poetry:
	poetry install
	touch .build_poetry

.install_torch: .build_poetry
	poetry run \
	pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
	touch .install_torch

.install_btcdet: .install_spconv
	poetry run \
	sh -c 'python3 setup.py develop'
	touch .install_btcdet

.clone_spconv:
	git clone -b v1.2.1  https://github.com/traveller59/spconv.git --recursive
	touch .clone_spconv

.install_spconv: .install_torch .clone_spconv
	poetry run \
	sh -c 'cd spconv&& python3 setup.py bdist_wheel && cd ./dist && pip3 install spconv-1.2.1-cp38-cp38-linux_x86_64.whl'
	touch .install_spconv


clean:
	rm -rf $(ENPTY_TARGETS)
	rm -rf spconv/dist
	rm -rf ./build
	rm ./poetry.lock
	poetry env list | cut -f1 -d' ' | while read name; do \
		poetry env remove "$$name"; \
	done

clean_all:
	rm -rf .clone_spconv
	rm -rf $(ENPTY_TARGETS)
	rm -rf spconv
	rm -rf ./build
	rm ./poetry.lock
	poetry env list | cut -f1 -d' ' | while read name; do \
		poetry env remove "$$name"; \
	done

list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'