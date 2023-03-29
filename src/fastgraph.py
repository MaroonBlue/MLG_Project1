import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patch

from threading import Thread
from sys import stdout
from time import sleep

from src.graph import Graph
from src.isomorphic_graphs import get_all_unique_graphs


class FastGraphSettings:
    def __init__(self, 
                 render = False, 
                 render_isomorphic_graphs = False, 
                 render_x_isomorphisms_per_column = 6, 
                 walk_size = 3) -> None:
        
        self.render = render
        self.render_isomorphic_graphs = render_isomorphic_graphs
        self.render_x_isomorphisms_per_column = render_x_isomorphisms_per_column
        self.walk_size = walk_size

class FastGraph:  
    def __init__(self, graph: Graph, settings: FastGraphSettings):

        self.render = settings.render
        self.render_iso = settings.render_isomorphic_graphs
        self.graph = graph
        self.walk_graphs = get_all_unique_graphs(settings.walk_size)
        self.walk_size = settings.walk_size
        self.walk_node_ids = []
        self.walk_points = {}

        if settings.render:
            figure = plt.figure(figsize=(10,10))
            figure.canvas.mpl_connect('key_press_event', self.on_key_press)
            figure.canvas.mpl_connect('button_press_event', self.on_mouse_press)

            grid_spec = gridspec.GridSpec(1, 1)
            if settings.render_isomorphic_graphs:
                div = settings.render_x_isomorphisms_per_column
                n = len(self.walk_graphs) // div + (len(self.walk_graphs) % div > 0)
                div_n = div + n

                grid = gridspec.GridSpecFromSubplotSpec(div, div_n, subplot_spec = grid_spec[0])
                axis = plt.subplot(grid[:,:div])
                for i, walk_graph in enumerate(self.walk_graphs):
                    sub_axis = plt.subplot(grid[i%div, div+i//div])
                    walk_graph.render(sub_axis, with_labels=False)
                    if not i:
                        sub_axis.set_title("Isomorphisms")
                        sub_axis.spines['bottom'].set(color = 'green', linewidth = 3)
                        sub_axis.spines['top'].set(color = 'green', linewidth = 3)
                        sub_axis.spines['left'].set(color = 'green', linewidth = 3)
                        sub_axis.spines['right'].set(color = 'green', linewidth = 3)
                    # else:
                    #     sub_axis.spines['bottom'].set_color('white')
                    #     sub_axis.spines['top'].set_color('white')
                    #     sub_axis.spines['left'].set_color('white')
                    #     sub_axis.spines['right'].set_color('white')
                    #     sub_axis.set_title(f"{i + 1}{'st' if i + 1 == 1 else 'nd' if i + 1 == 2 else 'th'} isomorphism")
                    figure.add_subplot(sub_axis)
            else:
                grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = grid_spec[0])
                axis = plt.subplot(grid[:,:])

            axis.set_title('X for step')
            axis.spines['bottom'].set_color('white')
            axis.spines['top'].set_color('white')
            axis.spines['left'].set_color('white')
            axis.spines['right'].set_color('white')
            graph.render(axis)
            figure.add_subplot(axis)

            self.figure = figure
            self.axis = axis
            plt.show()

    def start_auto_walk(self):
        t = Thread(target=self.auto_walk_loop)
        t.start()

    def auto_walk_loop(self):
        for i in range(1000):
            sleep(0.2)
            self.on_frame_update()

    def on_frame_update(self):
        if self.walk_node_ids:
            start = self.walk_node_ids.pop(0)
            self.walk_node_ids.append((start+self.walk_size) % len(self.graph.node_names))
        else:
            self.walk_node_ids = list(range(self.walk_size))
        self.draw_walk(self.walk_node_ids)

    def on_key_press(self, event):
        stdout.flush()
        if event.key == 'x':
            self.on_frame_update()
        elif event.key == 'c':
            self.start_auto_walk()

    def on_mouse_press(self, event):
        return
        # if event.inaxes:
        #     clickX = event.xdata
        #     clickY = event.ydata
        #     print(dir(event),event.key)
        #     if self.axis is None or self.axis==event.inaxes:
        #         annotes = []
        #         smallest_x_dist = float('inf')
        #         smallest_y_dist = float('inf')

        #         for x,y,a in self.data:
        #             if abs(clickX-x)<=smallest_x_dist and abs(clickY-y)<=smallest_y_dist :
        #                 dx, dy = x - clickX, y - clickY
        #                 annotes.append((dx*dx+dy*dy,x,y, a) )
        #                 smallest_x_dist=abs(clickX-x)
        #                 smallest_y_dist=abs(clickY-y)

        #         if annotes:
        #             annotes.sort() # to select the nearest node
        #             distance, x, y, annote = annotes[0]
        #             self.drawAnnote(event.inaxes, x, y, annote)

    # def drawAnnote(self, axis, x, y, annote):
    #     if (x, y) in self.drawnAnnotations:
    #         markers = self.drawnAnnotations[(x, y)]
    #         for m in markers:
    #             m.set_visible(not m.get_visible())
    #         self.axis.figure.canvas.draw()
    #     else:
    #         t = axis.text(x, y, "%s" % (annote), )
    #         m = axis.scatter([x], [y], marker='d', c='r', zorder=100)
    #         self.drawnAnnotations[(x, y)] = (t, m)
    #         self.axis.figure.canvas.draw()

    def draw_walk(self, node_ids):
        for node_id in self.walk_points.keys():
            text, marker = self.walk_points[node_id]
            text.set_visible(False)
            marker.set_visible(False)

        for node_id in node_ids:
            x, y = self.graph.node_name_position_map[node_id]
            text = self.axis.text(x, y, "%s" % (node_id), )
            marker = self.axis.scatter([x], [y], marker='*', c='r', zorder=2, s = 36)
            self.walk_points[node_id] = [text, marker]
            self.axis.figure.canvas.draw()