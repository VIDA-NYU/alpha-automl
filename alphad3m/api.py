from d3m_interface import AutoML as BaseAutoML


class AutoML(BaseAutoML):

    def __init__(self, output_folder, grpc_port=None, verbose=False):
        """Create/instantiate an AutoML object

        :param output_folder: Path to the output directory
        :param grpc_port: Port to be used by GRPC
        :param verbose: Whether or not to show all the logs from AutoML systems
        """

        automl_id = 'AlphaD3M'
        container_runtime = 'pypi'
        BaseAutoML.__init__(self, output_folder, automl_id, container_runtime, grpc_port, verbose)
