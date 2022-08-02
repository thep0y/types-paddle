#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:    thepoy
# @Email:     email@example.com
# @File Name: setup.py
# @Created:   2022-08-02 09:46:20
# @Modified:  2022-08-02 10:06:24

import codecs

from setuptools import setup


with codecs.open("README.md", "r", "utf-8") as fd:
    setup(
        name="types-paddle",
        version="0.0.1",
        description="""
        paddlepaddle 非官方类型库
        """,
        long_description_content_type="text/markdown",
        long_description=fd.read(),
        author="thepoy",
        author_email="thepoy@163.com",
        url="https://github.com/thep0y/types-paddle",
        license="MIT",
        keywords="paddle paddlepaddle 类型",
        packages=["paddle-stubs"],
        package_data={
            "paddle-stubs": [
                "fluid/__init__.pyi",
                "fluid/framework.pyi",
            ]
        },
    )
