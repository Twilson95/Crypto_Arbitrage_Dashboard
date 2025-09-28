import sys
import types


def _ensure_dummy_dash_modules():
    if "dash" not in sys.modules:
        dash_module = types.ModuleType("dash")
        dash_module.html = types.SimpleNamespace(
            Div=lambda *args, **kwargs: None,
            Label=lambda *args, **kwargs: None,
            H1=lambda *args, **kwargs: None,
            H2=lambda *args, **kwargs: None,
            Span=lambda *args, **kwargs: None,
        )
        dash_module.dcc = types.SimpleNamespace(
            Dropdown=lambda *args, **kwargs: None,
            Interval=lambda *args, **kwargs: None,
            Graph=lambda *args, **kwargs: None,
        )
        dash_module.__spec__ = types.SimpleNamespace()
        sys.modules["dash"] = dash_module

    if "dash_bootstrap_components" not in sys.modules:
        dbc_module = types.ModuleType("dash_bootstrap_components")
        dbc_module.__getattr__ = lambda name: (lambda *args, **kwargs: None)
        dbc_module.__spec__ = types.SimpleNamespace()
        sys.modules["dash_bootstrap_components"] = dbc_module

    if "river" not in sys.modules:
        river_module = types.ModuleType("river")
        river_module.__spec__ = types.SimpleNamespace()
        river_module.__path__ = []
        for submodule in [
            "compose",
            "preprocessing",
            "linear_model",
            "feature_extraction",
            "naive_bayes",
            "tree",
            "forest",
            "ensemble",
        ]:
            module_name = f"river.{submodule}"
            sub_module = types.ModuleType(module_name)
            sub_module.__spec__ = types.SimpleNamespace()
            sys.modules[module_name] = sub_module
            setattr(river_module, submodule, sub_module)
        sys.modules["river"] = river_module

    if "newsapi" not in sys.modules:
        newsapi_module = types.ModuleType("newsapi")

        class _DummyNewsApiClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def get_sources(self, *args, **kwargs):
                return {}

            def get_everything(self, *args, **kwargs):
                return {}

        newsapi_module.NewsApiClient = _DummyNewsApiClient
        newsapi_module.__spec__ = types.SimpleNamespace()
        sys.modules["newsapi"] = newsapi_module

        newsapi_exception_module = types.ModuleType("newsapi.newsapi_exception")

        class _DummyNewsAPIException(Exception):
            pass

        newsapi_exception_module.NewsAPIException = _DummyNewsAPIException
        newsapi_exception_module.__spec__ = types.SimpleNamespace()
        sys.modules["newsapi.newsapi_exception"] = newsapi_exception_module

    if "plotly" not in sys.modules:
        plotly_module = types.ModuleType("plotly")
        plotly_module.__spec__ = types.SimpleNamespace()
        plotly_module.__path__ = []

        class _DummyFigure:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        graph_objs_module = types.ModuleType("plotly.graph_objs")
        graph_objs_module.Figure = _DummyFigure
        graph_objs_module.__spec__ = types.SimpleNamespace()

        graph_objects_module = types.ModuleType("plotly.graph_objects")
        graph_objects_module.Figure = _DummyFigure
        graph_objects_module.__spec__ = types.SimpleNamespace()

        plotly_module.graph_objs = graph_objs_module
        plotly_module.graph_objects = graph_objects_module

        subplots_module = types.ModuleType("plotly.subplots")
        subplots_module.make_subplots = lambda *args, **kwargs: None
        subplots_module.__spec__ = types.SimpleNamespace()

        sys.modules["plotly"] = plotly_module
        sys.modules["plotly.graph_objs"] = graph_objs_module
        sys.modules["plotly.graph_objects"] = graph_objects_module
        sys.modules["plotly.subplots"] = subplots_module
        sys.modules["plotly.express"] = types.ModuleType("plotly.express")


_ensure_dummy_dash_modules()
