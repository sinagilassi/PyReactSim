# NOTE: COnfig
from .configs import __version__, __author__, __email__, __description__

# NOTE: app
from .app import create_batch_reactor, create_cstr_reactor, create_pfr_reactor, create_pbr_reactor

# NOTE: docs
from .docs.br import BatchReactor
from .docs.cstr import CSTRReactor
from .docs.pfr import PFRReactor
from .docs.pbr import PBRReactor

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    # app
    "create_batch_reactor",
    "create_cstr_reactor",
    "create_pfr_reactor",
    "create_pbr_reactor",
    # docs
    "BatchReactor",
    "CSTRReactor",
    "PFRReactor",
    "PBRReactor",
]
