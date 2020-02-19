# -*- coding: utf-8 -*-
"""
Author: cbing
Filename: LogManager.py
Date created: 2016-12-27 10:30
Last Modified: 2017-02-10 17:00
Modified by: cbing

Description:
    a simple wrapper of logging
Changelog:
    2016-12-27 12:00 create file

"""

import os
import logging
import traceback
import logging.handlers as LH
import platform

import conf


def log_compact_traceback(self):
    self.error(traceback.format_exc())


# log level, from high to low rate
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARN
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG

# log output type
STREAM = "stream"
SYSLOG = "syslog"
FILE = "file"


class LogManager(object):
    log_name = 'algo.log'
    created_filename = None
    created_modules = set()
    log_level = INFO
    log_handle = FILE
    auto_rotate_log = True

    @staticmethod
    def getFileHandler(filePath):
        if LogManager.auto_rotate_log:
            handler = logging.handlers.TimedRotatingFileHandler(
                filePath,
                when='midnight',
                backupCount=7,
                encoding='utf8'
            )
        else:
            handler = logging.FileHandler(filePath, encoding='utf8')
        return handler

    @staticmethod
    def get_logger(moduleName, log_relative_path=conf.log_dir):
        # If we have it already, return it directly
        log_name = moduleName + '.log'
        if (moduleName in LogManager.created_modules):
            return logging.getLogger(moduleName)
        logger = logging.getLogger(moduleName)
        logger.log_last_except = log_compact_traceback
        logger.setLevel(LogManager.log_level)

        def get_log_path():
            if LogManager.created_filename is None:
                log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_relative_path)
                # LogManager.created_filename = os.path.join(log_dir, LogManager.log_name)
                LogManager.created_filename = os.path.join(log_dir, log_name)
            return LogManager.created_filename

        # create handler
        if LogManager.log_handle == SYSLOG:
            if platform.system() == 'Linux':
                # debug logs use LOG_LOCAL1
                ch = LH.SysLogHandler('/dev/log', facility=LH.SysLogHandler.LOG_LOCAL1)
            else:
                ch = LogManager.getFileHandler(get_log_path())
        elif LogManager.log_handle == FILE:
            ch = LogManager.getFileHandler(get_log_path())
        else:
            ch = logging.StreamHandler()

        ch.setLevel(LogManager.log_level)
        # create formatter and add it to the handlers
        formatlist = ['%(asctime)s', '%(name)s', '%(levelname)s', '%(message)s']
        formatter = logging.Formatter(' - '.join(formatlist))
        ch.setFormatter(formatter)
        # add the handlers to logger  
        logger.addHandler(ch)
        LogManager.created_modules.add(moduleName)
        return logger

    @staticmethod
    def set_log_level(lv):
        LogManager.log_level = lv

    @staticmethod
    def set_log_handle(handle):
        LogManager.log_handle = handle

    @staticmethod
    def set_log_name(log_name):
        LogManager.log_name = log_name


def main():
    # "application" code  
    logger = LogManager.get_logger("LogManager.Main")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")
    LogManager.get_logger("Test").debug("train_data debug message")

    try:
        raise Exception("A")
    except Exception as e:
        logger.log_last_except(e)


