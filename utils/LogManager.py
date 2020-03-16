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


# def initialize_logging(name):
#     logger = logging.getLogger(name)
#     logger.setLevel(level=logging.INFO)
#     handler = logging.FileHandler("output/log/%s.log" % name)
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     logger.addHandler(handler)
#     logger.addHandler(console)
#     return logger

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
FILE_AND_TERMINAL = "file_and_termianl"


class LogManager(object):
    log_name = 'algo.log'
    created_filename = None
    created_modules = set()
    log_level = INFO
    log_handle = FILE_AND_TERMINAL
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
    def get_logger(moduleName, log_relative_path=conf.LOG_DIR):
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

        formatlist = ['%(asctime)s', '%(name)s', '%(levelname)s', '%(message)s']
        formatter = logging.Formatter(' - '.join(formatlist))
#         if LogManager.log_handle ==FILE_AND_TERMINAL:  # create handler, output the msg in terminal and log at the meantime
        fh = LogManager.getFileHandler(get_log_path())
        ch = logging.StreamHandler()
        ch.setLevel(LogManager.log_level)
        fh.setLevel(LogManager.log_level)
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
#         else:  # just output to terminal
#             ch = logging.StreamHandler()
#             ch.setLevel(LogManager.log_level)
#             ch.setFormatter(formatter)
#             logger.addHandler(ch)
            
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


