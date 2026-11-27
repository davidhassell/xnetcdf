"""Utilities for the `earthkit` backend."""

from collections.abc import Mapping

from .utils_general import get_dataset_name_and_protocol, get_library

_SECTIONS = ("from_source", "profile", "mapping")
_MAPPING_KEYS = ("calendar", "decreasing_atmos_pressure", "squeeze")


def earthkit_parse_options(options):
    """Parse the ``earthkit_options`` into keywords for `grfive.File`.

    The options are in separate sections, so that the keywords of
    `earthkit.data.from_source`, of the `earthkit.data` profile, and of
    the CF mapping can not clash. The input is never modified.

    :Parameters:

        options: `dict` or `None`
            The ``earthkit_options``, with any of the sections
            ``'from_source'``, ``'profile'`` and ``'mapping'``, and
            optionally ``mode='r'``. See `xnetcdf.Dataset`.

    :Returns:

        `dict`
            Keyword arguments for `grfive.File`.

    """
    options = dict(options or {})

    mode = options.pop("mode", "r")
    if mode != "r":
        raise ValueError(f"Can't set mode={mode!r} in earthkit_options")

    unknown = sorted(set(options).difference(_SECTIONS), key=str)
    if unknown:
        raise ValueError(
            f"Invalid earthkit_options key(s) {unknown}. Valid keys "
            f"are {_SECTIONS}"
        )

    kwargs = {}

    # Keywords for earthkit.data.from_source. The source type, which
    # is its first positional argument, is 'name'.
    source = _section(options, "from_source")
    if "name" in source:
        kwargs["source_type"] = source.pop("name")

    kwargs["source_kwargs"] = source

    # Keywords for the earthkit.data profile. The profile, which is
    # its first positional argument, is 'name'. A profile name or
    # object may be given instead of a dictionary.
    profile = options.get("profile")
    if profile is not None and not isinstance(profile, Mapping):
        profile = {"name": profile}

    profile = _section({"profile": profile}, "profile")
    if "name" in profile:
        kwargs["profile"] = profile.pop("name")

    if "squeeze" in profile:
        raise ValueError(
            "Set 'squeeze' in the 'mapping' section of earthkit_options, "
            "not in 'profile'"
        )

    kwargs["profile_kwargs"] = profile

    # Keywords for the mapping to CF
    mapping = _section(options, "mapping")
    unknown = sorted(set(mapping).difference(_MAPPING_KEYS))
    if unknown:
        raise ValueError(
            f"Invalid earthkit_options['mapping'] key(s) {unknown}. "
            f"Valid keys are {_MAPPING_KEYS}"
        )

    kwargs.update(mapping)

    return kwargs


def _section(options, name):
    """Return a copy of an `earthkit_options` section."""
    section = options.get(name)
    if section is None:
        return {}

    if not isinstance(section, Mapping):
        raise TypeError(
            f"earthkit_options[{name!r}] must be a dictionary. "
            f"Got {section!r}"
        )

    return dict(section)


def earthkit_read(dataset, options):
    """Read a dataset with `earthkit`.

    :Parameters:

        dataset:
            The definition of the dataset to be read. One of:

            * string-like (such as `str` or `pathlib.Path`)
            * file-like (such as `io.BufferedReader` or the result
                         of an `fsspec` file system open)

             An exception is raised if the *dataset* can't be
             interpreted.

        options: `dict`
            The ``earthkit_options`` of `xnetcdf.Dataset`, in the
            sections described by `earthkit_parse_options`.

    :Returns:

        (`grfive.File`, `dict`, library)
            The opened dataset, the dataset's global attributes, and
            the `earthkit` library itself.

    """
    import earthkit.data as ekd

    from .grfive import File

    kwargs = earthkit_parse_options(options)

    dataset_name = ""
    protocol = -1

    if isinstance(dataset, ekd.FieldList):
        nc = File(dataset, mode="r", **kwargs)
        library = get_library(dataset)
        # Not all field lists have a file name
        dataset_name = getattr(dataset, "path", None) or ""
        owns_accessor = False

        # Attempt to get the dataset name and file system protocol
        try:
            # fsspec file-like
            dataset_name = dataset._fh.path
        except AttributeError:
            try:
                # BinaryIO
                dataset_name = dataset._fh.name
            except AttributeError:
                pass
            else:
                # BinaryIO
                protocol = "file"
        else:
            try:
                # fsspec file-like
                protocol = dataset._fh.fs.protocol
            except AttributeError:
                pass

        if not dataset_name:
            dataset_name = "<earthkit-like>"

    else:
        import earthkit

        dataset, dataset_name, protocol = get_dataset_name_and_protocol(
            dataset
        )
        nc = File(dataset, mode="r", **kwargs)

        library = earthkit
        owns_accessor = True

    return {
        "dataset_name": dataset_name,
        "protocol": protocol,
        "nc": nc,
        "attrs": nc.attrs,
        "backend_api": "pyfive",
        "library": library,
        "owns_accessor": owns_accessor,
    }
