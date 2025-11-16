"""
Base LLM Agent Module

Abstract base class for LLM agents supporting different backends
(OpenAI, HuggingFace, etc.)
"""

from abc import ABC

import structlog

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for LLM agents"""
    CATEGORY_DESCRIPTIONS = {
                "Bug": (
                    "A report describing an error, malfunction, or incorrect behavior in a Boost library. "
                    "These messages often include details like reproduction steps, logs, or requests for fixes."
                ),
                "Design": (
                    "A discussion about how a Boost library or feature should be structured, organized, or implemented. "
                    "It focuses on architecture, API choices, performance trade-offs, or conceptual improvements."
                ),
                "Review": (
                    "An evaluation of a proposed Boost library or version by community members or reviewers. "
                    "It summarizes feedback, acceptance decisions, and suggestions for improvement."
                ),
                "Security": (
                    "A report or notice about vulnerabilities, unsafe functions, or security risks in Boost code. "
                    "It may include patches, mitigation steps, or references to CVE issues."
                ),
            }
    KNOWN_LIBRARIES = [
        'Accumulators', 'Algorithm', 'Align', 'Any', 'Array', 'Asio', 'Assert',
        'Assign', 'Atomic', 'Awaitables', 'Beast', 'Bimap', 'Bind', 'Bloom',
        'Call Traits', 'CallableTraits', 'CharConv', 'Chrono', 'Circular Buffer',
        'Cobalt', 'Compat', 'Compressed Pair', 'Compute', 'Concept Check', 'Config',
        'Container', 'Container Hash', 'Context', 'Contract', 'Conversion', 'Convert',
        'Core', 'Coroutine', 'Coroutine2', 'CRC', 'Date Time', 'Describe', 'Detail',
        'DLL', 'Dynamic Bitset', 'Enable If', 'Endian', 'Exception', 'Fiber',
        'Filesystem', 'Flyweight', 'Foreach', 'Format', 'Function', 'Function Types',
        'Functional', 'Fusion', 'Geometry', 'GIL', 'Graph',
        'GraphParallel', 'Hana', 'Hash2', 'Heap', 'Histogram', 'HOF', 'ICL',
        'Identity Type', 'In Place Factory, Typed In Place Factory', 'Integer',
        'Interprocess', 'Interval', 'Intrusive', 'IO', 'Iostreams', 'Iterator',
        'JSON', 'Lambda', 'Lambda2', 'LEAF', 'Lexical Cast', 'Local Function',
        'Locale', 'Lockfree', 'Log', 'Math', 'Member Function', 'Meta State Machine',
        'Metaparse', 'Min Max', 'Move', 'Mp11', 'MPI', 'MPL', 'MQTT5', 'MultiArray',
        'MultiIndex', 'Multiprecision', 'MySQL', 'Nowide', 'Numeric Conversion',
        'Odeint', 'Operators', 'Optional', 'Outcome', 'Parameter',
        'Parameter Python Bindings', 'Parser', 'PFR', 'Phoenix', 'Pointer Container',
        'PolyCollection', 'Polygon', 'Pool', 'Predef', 'Preprocessor', 'Process',
        'Program Options', 'Property Map', 'Property Map Parallel', 'Property Tree',
        'Proto', 'Python', 'QVM', 'Random', 'Range', 'Ratio', 'Rational', 'Redis',
        'Ref', 'Regex', 'Result Of', 'Safe Numerics', 'Scope', 'Scope Exit',
        'Serialization', 'Signals2', 'Smart Ptr', 'Sort', 'Spirit', 'Stacktrace',
        'Statechart', 'Static Assert', 'Static String', 'Stl_interfaces',
        'String Algo', 'String View', 'Swap', 'System', 'Test', 'Thread',
        'ThrowException', 'Timer', 'Tokenizer', 'Tribool', 'TTI', 'Tuple',
        'Type Erasure', 'Type Index', 'Type Traits', 'Typeof', 'uBLAS', 'Units',
        'Unordered', 'URL', 'Utility', 'Uuid', 'Value Initialized', 'Variant',
        'Variant2', 'VMD', 'Wave', 'WinAPI', 'Xpressive', 'YAP',
    ]

    def __init__(self):
        self.logger = logger.bind(component=self.__class__.__name__)
