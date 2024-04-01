# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:17:06 2018

@author: Thomas
"""

from igraph import Graph

# Class which can have attributes set.
class expando(object):
    pass

class MGraph(Graph):
    def __init__(self, *args, **kwds): 
        Graph.__init__(self, *args, **kwds)
        
        self.lines = expando()
        self.lines.default_size = 2
        self.lines.max_size = 8
        self.lines.min_size = .1 
        self.trade_threshold = 1 # in % of biggest trade
        self.lines.color = 'rgb(100,100,100)'
        self.lines.opacity = 0.5
        
        self.markers = expando()
        self.markers.default_size = 12
        self.markers.default_line_size = 0
        self.markers.colorscale = 'Jet'
        self.markers.color = 'rgb(204,153,0)'
        self.markers.line_color = 'rgb(50,50,50)'
        self.markers.symbol = 'circle'
        
        self.BuildGraphOfMarketGraph(True)
        return
    
    @classmethod 
    def Load(cls, f, format=None, *args, **kwds):
        g = Graph.Load(f, format=format, *args, **kwds)
        out = MGraph(directed=True)
        out.add_vertices(len(g.vs))
        for i in range(len(g.vs)):
            for idx, k in enumerate(g.vs[i].attributes()):
                out.vs[i][k] = g.vs[i][k]
        for i in range(len(g.es)):
            out.add_edge(g.es[i].source, g.es[i].target)
            for idx, k in enumerate(g.es[i].attributes()):
                out.es[i][k] = g.es[i][k]
        return out
    
    @classmethod 
    def Save(cls, g, f, format=None):
        out = Graph(directed=True)
        out.add_vertices(len(g.vs))
        for i in range(len(g.vs)):
            for idx, k in enumerate(g.vs[i].attributes()):
                out.vs[i][k] = g.vs[i][k]
        for i in range(len(g.es)):
            out.add_edge(g.es[i].source, g.es[i].target)
            for idx, k in enumerate(g.es[i].attributes()):
                out.es[i][k] = g.es[i][k]
        return out.save(f, format=format)
    
    def BuildLayout(self, first=False):
        update = False
        if first == True:
            self.N_vertices = len(self.vs)
            self.N_edges = len(self.es)
        if first or self.N_vertices != len(self.vs) or self.N_edges != len(self.es):
            update = True
            self.N_vertices = len(self.vs)
            self.layt = self.layout()
            self.Xn = [self.layt[k][0] for k in range(self.N_vertices)]
            self.Yn = [self.layt[k][1] for k in range(self.N_vertices)]
        if first or (self.N_vertices == len(self.vs) and self.N_edges != len(self.es)):
            update = True
            self.N_edges = len(self.es)
            self.Xe = []
            self.Ye = []
            for e in self.es:
                self.Xe += [self.layt[e.source][0], self.layt[e.target][0], None]
                self.Ye += [self.layt[e.source][1], self.layt[e.target][1], None]
        return update
    
    def BuildGraphOfMarketGraph(self, first=False):
        if self.BuildLayout(first):
            return True
        return False
    
    def BuildEdges(self, trades=None):
        if trades is not None:
            t_norm = abs(trades).max()
            traces = []
            for e in self.es:
                l_width = trades[e.source][e.target] * self.lines.max_size / t_norm
                if self.lines.min_size <= l_width:
                    traces.append((e.source, e.target, l_width))
            return traces
        else:
            return []
    
    def UpdateGraphEdges(self, update=False, trades=None):
        if update:
            traces = self.BuildEdges(trades)
            return traces
        return []
