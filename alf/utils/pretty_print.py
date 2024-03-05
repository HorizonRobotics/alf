# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import StringIO
import pprint
try:
    from pygments import highlight
    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import PythonLexer
    _py_color = True
except:
    # fallback to no python syntax highlighting
    _py_color = False


class PrettyPrinter(pprint.PrettyPrinter):
    """Copied from https://stackoverflow.com/questions/30062384/pretty-print-namedtuple
    """

    def format_namedtuple(self, object, stream, indent, allowance, context,
                          level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write(object.__class__.__name__ + '(')
        object_dict = object._asdict()
        length = len(object_dict)
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            inline_stream = StringIO()
            PrettyPrinter.format_namedtuple_items(
                self,
                object_dict.items(),
                inline_stream,
                indent,
                allowance + 1,
                context,
                level,
                inline=True)
            max_width = self._width - indent - allowance
            if len(inline_stream.getvalue()) > max_width:
                PrettyPrinter.format_namedtuple_items(
                    self,
                    object_dict.items(),
                    stream,
                    indent,
                    allowance + 1,
                    context,
                    level,
                    inline=False)
            else:
                stream.write(inline_stream.getvalue())
        write(')')

    def format_namedtuple_items(self,
                                items,
                                stream,
                                indent,
                                allowance,
                                context,
                                level,
                                inline=False):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ', '
        else:
            delimnl = ',\n' + ' ' * indent
            write('\n' + ' ' * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key + '=')
            self._format(ent, stream, indent + len(key) + 2,
                         allowance if last else 1, context, level)
            if not last:
                write(delimnl)

    def _format(self, object, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if (hasattr(object, '_asdict')
                and type(object).__repr__ not in self._dispatch):
            self._dispatch[type(object).
                           __repr__] = PrettyPrinter.format_namedtuple
        super()._format(object, stream, indent, allowance, context, level)


pp = PrettyPrinter(indent=2)


def pformat_pycolor(obj):
    if _py_color:
        return highlight(
            pp.pformat(obj), PythonLexer(), Terminal256Formatter())
    return pp.pformat(obj)
