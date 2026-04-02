# NOTE: COnfig
from .configs import __version__, __author__, __email__, __description__

# NOTE: app
from .app import create_batch_reactor, create_cstr_reactor

# NOTE: docs
from .docs.br import BatchReactor
from .docs.cstr import CSTRReactor

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    # app
    "create_batch_reactor",
    "create_cstr_reactor",
    # docs
    "BatchReactor",
    "CSTRReactor",
]
