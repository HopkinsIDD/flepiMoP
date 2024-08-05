import click


@click.command()
@click.option('--aws', is_flag=True, help='push files to aws')
@click.option('--local', is_flag=True,)
@click.option('--flepi_run_index', 'flepi_run_index', envvar='FLEPI_RUN_INDEX', type=click.STRING, required=True)
@click.option('--flepi_prefix', 'flepi_prefix', envvar='FLEPI_PREFIX', type=click.STRING, required=True)
def flepimop_push(flepi_run_index,
                  ):
    if aws:
        push_to_aws(input_files)
    else:
        move_local(input_files)