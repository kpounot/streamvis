from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, DataTable, NumberFormatter, TableColumn

import streamvis as sv

receiver = sv.receiver
doc = curdoc()
doc.title = f"{sv.page_title} Statistics"

table_columns = [
    TableColumn(field='run_names', title="Run Name"),
    TableColumn(field='nframes', title="Total Frames"),
    TableColumn(field='bad_frames', title="Bad Frames"),
    TableColumn(field='sat_pix_nframes', title="Sat pix frames"),
    TableColumn(field='laser_on_nframes', title="Laser ON frames"),
    TableColumn(field='laser_on_hits', title="Laser ON hits"),
    TableColumn(
        field='laser_on_hits_ratio',
        title="Laser ON hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
    TableColumn(field='laser_off_nframes', title="Laser OFF frames"),
    TableColumn(field='laser_off_hits', title="Laser OFF hits"),
    TableColumn(
        field='laser_off_hits_ratio',
        title="Laser OFF hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
]

table_source = ColumnDataSource(receiver.stats.data)
table = DataTable(source=table_source, columns=table_columns, height=50, index_position=None)

sum_table_source = ColumnDataSource(receiver.stats.sum_data)
sum_table = DataTable(
    source=sum_table_source, columns=table_columns, height=50, index_position=None,
)


# update statistics callback
def update_statistics():
    table_source.data = receiver.stats.data
    sum_table_source.data = receiver.stats.sum_data


# reset statistics button
def reset_stats_button_callback():
    receiver.stats.reset()


reset_stats_button = Button(label="Reset Statistics", button_type='default')
reset_stats_button.on_click(reset_stats_button_callback)

layout = column(
    column(table, sizing_mode="stretch_both"),
    sum_table,
    reset_stats_button,
    sizing_mode="stretch_width",
)

doc.add_root(layout)
doc.add_periodic_callback(update_statistics, 1000)