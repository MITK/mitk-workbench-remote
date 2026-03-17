# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or find it in LICENSE.txt.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exception hierarchy for mitk-workbench-remote.

All exceptions inherit from MitkError. REST error codes are mapped to
specific exception types by RestTransport.
"""


class MitkError(Exception):
    """Base class for all mitk-workbench-remote exceptions."""


class MitkConnectionError(MitkError):
    """Raised when the network is unreachable or the connection is refused."""


class AuthenticationError(MitkError):
    """Raised on HTTP 401 (unauthenticated) or 403 (unauthorized) responses.

    Callers can inspect ``status_code`` to distinguish between a missing or
    invalid token (401) and a valid token with insufficient permissions (403).

    Args:
        status_code: The HTTP status code (401 or 403).
        message: The human-readable error message from the response body.
    """

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class NodeNotFoundError(MitkError):
    """Raised when the server returns NODE_NOT_FOUND (HTTP 404).

    Args:
        uid: The REST API UID of the node that was not found. May be an empty string
            at the transport layer when no UID context is available.
    """

    def __init__(self, uid: str) -> None:
        self.uid = uid
        super().__init__(f"Node not found: {uid!r}" if uid else "Node not found")


class StaleNodeError(MitkError):
    """Raised when a DataNode reference is used after the node was deleted."""


class DataStorageNotAvailableError(MitkError):
    """Raised when the server returns DATASTORAGE_NOT_AVAILABLE (HTTP 503)."""


class UnsupportedDataTypeError(MitkError):
    """Raised when a node's data type cannot be represented in Python.

    The MITK Workbench may hold data types (e.g. Surface, PointSet) for which
    this library has no in-memory representation yet. Use
    :meth:`~mitk_workbench_remote.node.DataNode.save_data` to download the raw
    bytes to a file instead.

    Args:
        data_type: The MITK data type string that is not supported.
    """

    def __init__(self, data_type: str) -> None:
        self.data_type = data_type
        super().__init__(
            f"Data type {data_type!r} is not supported for in-memory representation. "
            f"Use save_data() to download the raw bytes to a file."
        )


class TransferError(MitkError):
    """Raised on UNSUPPORTED_FORMAT (HTTP 415) or I/O failure during transfer."""


class RenderingError(MitkError):
    """Raised when the server returns RENDERING_ERROR (HTTP 422)."""


class ApiError(MitkError):
    """Catch-all for 4xx/5xx responses not covered by a more specific type.

    Args:
        status_code: The HTTP status code.
        code: The error code string from the RFC 7807 body (e.g. ``"UNKNOWN"``).
        message: The human-readable error message from the response body.
            Stored on ``.message`` for structured access; ``str(err)`` returns
            the full ``"HTTP <status> [<code>]: <message>"`` form.
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"HTTP {status_code} [{code}]: {message}")
