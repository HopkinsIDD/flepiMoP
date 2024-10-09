
import click
import warnings

# https://stackoverflow.com/a/50402799
# CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/
class DeprecatedOption(click.Option):

    def __init__(self, *args, **kwargs):
        self.deprecated = kwargs.pop('deprecated', ())
        self.preferred = kwargs.pop('preferred', args[0][-1])
        super(DeprecatedOption, self).__init__(*args, **kwargs)

class DeprecatedOptionsCommand(click.Command):

    def make_parser(self, ctx):
        """Hook 'make_parser' and during processing check the name
            used to invoke the option to see if it is preferred"""

        parser = super(DeprecatedOptionsCommand, self).make_parser(ctx)

        # get the parser options
        options = set(parser._short_opt.values())
        options |= set(parser._long_opt.values())

        for option in options:
            if not isinstance(option.obj, DeprecatedOption):
                continue

            def make_process(an_option):
                """ Construct a closure to the parser option processor """

                orig_process = an_option.process
                deprecated = getattr(an_option.obj, 'deprecated', None)
                preferred = getattr(an_option.obj, 'preferred', None)
                msg = "Expected `deprecated` value for `{}`"
                assert deprecated is not None, msg.format(an_option.obj.name)

                def process(value, state):
                    """The function above us on the stack used 'opt' to
                        pick option from a dict, see if it is deprecated """

                    # reach up the stack and get 'opt'
                    import inspect
                    frame = inspect.currentframe()
                    try:
                        opt = frame.f_back.f_locals.get('opt')
                    finally:
                        del frame

                    if opt in deprecated:
                        msg = "'{}' has been deprecated, use '{}'"
                        warnings.warn(msg.format(opt, preferred),
                                      FutureWarning)

                    return orig_process(value, state)

                return process

            option.process = make_process(option)

        return parser
