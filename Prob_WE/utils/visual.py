"""
    Author: Marco Maggipinto
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

from visdom import Visdom
import time
import torch

class VisualManager:
    """Encapsulates a visdom object to perform plotting operations on a specific environment"""

    def __init__(self, vis: Visdom, env, default_plot_type='line', log_filename=None,
                 opts_dict: dict = {}):
        """
        Constructor
        :param vis: a Visdom object
        :param env: the environment name
        :param default_plot_type: the default plot type that is used when a plotting fuction is called, default:'line'
        :param log_filename: file to log operations
        :param opts_dict: a dictionary containing the options for each plot type e.g.
                {'globals': global_opts','line': line_opts, 'scatter': scatter_opts}
        """
        self.vis = vis
        self.__check_connection()
        self.env = env
        self.default_plot_type = default_plot_type
        self.log_filename = log_filename
        self.opts_dict = opts_dict

    def __get_plot_type(self, plot_type):
        return plot_type if plot_type else self.default_plot_type

    def __get_opts(self, plot_type, title, opts):
        if not opts:
            opts = self.opts_dict.get(plot_type, {})
            global_ots = self.opts_dict.get('globals', {})
            opts = {**global_ots, **opts}
        if title:
            opts['title'] = title
        return opts

    def __check_connection(self):
        startup_sec = 1
        while not self.vis.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
        assert self.vis.check_connection(), 'No connection could be formed quickly'

    def __update_win(self, *args, win, plot_type=None, update='append', title=None, opts=None):
        plot_type = self.__get_plot_type(plot_type)
        f = getattr(self.vis, plot_type)
        opts = self.__get_opts(plot_type, title, opts)
        if self.vis.win_exists(win, env=self.env):
            win = f(*args, win=win, env=self.env, opts=opts, update=update)
        else:
            win = self.plot(*args, plot_type=plot_type)
        return win

    def plot(self, *args, plot_type=None, title=None, opts=None):
        """
        Plots data provided into args
        :param args: arguments that are passed the plotting function: don't use names usually
        :param plot_type: plot type: 'line', 'scatter', etc.
        :param title: title of the window, by default it is used the one in the options provided in the constructor
        :param opts: custom options that has priority over the options provided in the constrctor
        :return: window id
        """
        plot_type = self.__get_plot_type(plot_type)
        f = getattr(self.vis, plot_type)
        opts = self.__get_opts(plot_type, title, opts)
        win = f(*args, env=self.env, opts=opts)
        return win

    def update(self, *args, win, plot_type=None, title=None, opts=None):
        """
        Appends data provided into args to the specified window, if the window does not exist it creates one
        :param args: arguments that are passed the plotting function: don't use names usually
        :param win: thw window where to append the data
        :param plot_type: plot type: 'line', 'scatter', etc.
        :param title: title of the window, by default it is used the one in the options provided in the constructor
        :param opts: custom options that has priority over the options provided in the constrctor
        :return: window id
        """
        return self.__update_win(*args, win=win, plot_type=plot_type, title=title, opts=opts)

    def replace(self, *args, win, plot_type=None, title=None, opts=None):
        """
        Replaces data provided into args to the specified window, if the window does not exist it creates one
        :param args: arguments that are passed the plotting function: don't use names usually
        :param win: thw window where to replace the data
        :param plot_type: plot type: 'line', 'scatter', etc.
        :param title: title of the window, by default it is used the one in the options provided in the constructor
        :param opts: custom options that has priority over the options provided in the constrctor
        :return: window id
        """
        return self.__update_win(*args, win=win, plot_type=plot_type, update='replace', title=title, opts=opts)

    def call_function(self, *args, function_name):
        f = getattr(self.vis, function_name)
        return f(*args, env=self.env)

    def save(self):
        """
        Saves the environment
        """
        self.vis.save([self.env])


class Logger:
    win = []

    def __init__(self, vm: VisualManager, title, opt, plot_type='line'):
        self.vm = vm
        self.title = title
        self.opt = opt
        self.plot_type = plot_type

    def log(self, data, title=None):
        if title is None:
            title = self.title
        self.vm.call_function(self.win, function_name='close')
        self.win = self.vm.plot(data, plot_type=self.plot_type, opts=self.opt, title=title)


class LineLogger(Logger):
    win = []
    last_value = []
    idx = 0

    def __init__(self, vm, title, log_period, opt):
        super(LineLogger, self).__init__(vm, title, opt)
        self.log_period = log_period

    def update(self, value, title=None):
        if title is None:
            title = self.title
        if self.idx != 0:
            x = torch.Tensor([self.idx - self.log_period, self.idx])
            y = torch.Tensor([self.last_value, value])
            self.win = self.vm.update(y, x, win=self.win, title=title)
        self.last_value = value
        self.idx += self.log_period

