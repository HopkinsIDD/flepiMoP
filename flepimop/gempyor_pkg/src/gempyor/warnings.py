"""Custom warnings for the `gempyor` package."""

__all__ = ("ConfigurationWarning",)


class ConfigurationWarning(UserWarning):
    """
    A warning for potential configuration issues.

    This warning indicates to users that there is a potential issue with their
    configuration file. While this does not indicate a critical issue and `flepiMoP`
    can continue processing it this warning does let the user know that their
    configuration file should be fixed due to potential unexpected results.

    Examples:
        >>> import warnings
        >>> import pytest
        >>> from gempyor.warnings import ConfigurationWarning
        >>> with pytest.warns(ConfigurationWarning) as record:
        ...     warnings.warn(
        ...         "There is an issue with your config file!",
        ...         ConfigurationWarning,
        ...     )
        >>> assert len(record) == 1
        >>> str(record[0].message)
        'There is an issue with your config file!'
    """
