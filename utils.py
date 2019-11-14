
def save_graph(execute_callback, **args):
    logdir = 'logs/func/graph'
    writer = tf.summary.create_file_writer(logdir)


    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    r = execute_callback(**args)

    with writer.as_default():
        tf.summary.trace_export(
          name="my_func_trace",
          step=0,
          profiler_outdir=logdir)

    return r
