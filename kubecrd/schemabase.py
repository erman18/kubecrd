import datetime
from enum import Enum
import json
import logging
import re
from typing import Optional, Type, get_type_hints

import kubernetes
import yaml
from apischema import (
    serialize as apischema_serialize,
    deserialize as apischema_deserialize,
)
from apischema.json_schema import deserialization_schema
from kubernetes import client as k8s_sync_client
from kubernetes import config, utils
from kubernetes.client.models.v1_object_meta import V1ObjectMeta
from kubernetes_asyncio import client as k8s_async_client
from kubernetes_asyncio import config as async_config

# ObjectMeta_attribute_map is simply the reverse of the
# V1ObjectMeta.attribute_map , which is a mapping from python attribute to json
# key while this is the opposite from json key to python attribute so that we
# can pass in the values to instantiate the V1ObjectMeta object.
ObjectMeta_attribute_map = {
    value: key for key, value in V1ObjectMeta.attribute_map.items()
}

logger = logging.getLogger(__name__)


def to_camel_case(snake_str):
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # and join them all together.
    return components[0] + "".join(x.title() for x in components[1:])


def to_snake_case(camel_str):
    """Convert camelCase to snake_case.

    :param camel_str: String in camelCase format
    :type camel_str: str
    :returns: String in snake_case format
    :rtype: str
    """
    # Use regex to find uppercase letters that are not at the beginning of the string
    # and replace them with an underscore followed by the lowercase letter.
    # Also handles the case of consecutive uppercase letters.
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def safe_to_snake_case(camel_str, cls=None, context=""):
    """Convert camelCase to snake_case with validation against class fields.

    :param camel_str: String in camelCase format
    :param cls: Optional class to validate field existence
    :param context: Context for error message
    :returns: String in snake_case format
    :raises: AttributeError if the converted field doesn't exist on cls
    """
    snake_str = to_snake_case(camel_str)

    # If class is provided, validate the field exists
    if cls is not None:
        # Get all attributes of the class
        if hasattr(cls, "__annotations__") and hasattr(cls.__annotations__, context):
            # For dataclasses, check annotations
            if snake_str not in cls.__annotations__.context:
                raise AttributeError(
                    f"Field '{camel_str}' (converted to '{snake_str}') "
                    f"does not exist on {cls.__name__} in {context} "
                    f"({cls.__annotations__.context})"
                )
        elif hasattr(cls, "__slots__"):
            # For classes with __slots__
            if snake_str not in cls.__slots__:
                raise AttributeError(
                    f"Field '{camel_str}' (converted to '{snake_str}') "
                    f"does not exist on {cls.__name__} {context}"
                )

    return snake_str


def get_attr_class(parent_class: Type, attr_name: str) -> Optional[Type]:
    """
    Get the type annotation of a class attribute without instantiating the parent.

    Args:
        parent_class: The class to inspect.
        attr_name: The attribute name to look up.

    Returns:
        The type class of the attribute or None if not found/typed.
    """
    try:
        # Try to get fully resolved type hints
        hints = get_type_hints(parent_class)
        spec_type = hints.get(attr_name)
        if spec_type is not None:
            return spec_type

        # Fallback to class annotations dictionary
        if hasattr(parent_class, "__annotations__"):
            annotation = parent_class.__annotations__.get(attr_name)
            if isinstance(annotation, type):
                return annotation

        # Check if it's a class variable with a default value
        if attr_name in vars(parent_class):
            attr_value = getattr(parent_class, attr_name)
            if attr_value is not None:
                return type(attr_value)

        return None
    except (TypeError, AttributeError):
        # Handle error cases where type resolution fails
        return None


def get_k8s_client(provided=None) -> k8s_sync_client.ApiClient:
    if provided:
        return provided
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return k8s_sync_client.ApiClient()


async def get_k8s_async_client(provided=None) -> k8s_async_client.ApiClient:
    if provided:
        return provided
    try:
        async_config.load_incluster_config()
    except async_config.ConfigException:
        await async_config.load_kube_config()
    return k8s_async_client.ApiClient()


class KubeResourceBase:
    """KubeResourceBase is base class that provides methods to converts dataclass
    into Kubernetes CR. It provides ability to create a Kubernetes CRD from the
    class and supports deserialization of the object JSON from K8s into Python
    obects with support for Metadata.
    """

    @classmethod
    def apischema(cls):
        """Get serialized openapi 3.0 schema for the cls.

        The output is a dict with (possibly nested) key-value pairs based on
        the schema of the class. This is used to generate the CRD schema down
        the line which rely on (a subset?) of OpenAPIV3 schema for the
        definition of a Kubernetes Custom Resource.
        """
        use_camel = getattr(cls, "__camel_case__", True)
        return deserialization_schema(
            cls,
            all_refs=False,
            additional_properties=True,
            with_schema=False,
            aliaser=to_camel_case if use_camel else None,
        )

    @classmethod
    def apischema_json(cls):
        """JSON Serialized OpenAPIV3 schema for the cls."""
        return json.dumps(cls.apischema())

    @classmethod
    def apischema_yaml(cls):
        """YAML Serialized OpenAPIV3 schema for the cls."""
        # return yaml.dump(cls.apischema(), Dumper=yaml.Dumper)
        yaml_schema = yaml.load(cls.apischema_json(), Loader=yaml.Loader)
        return yaml.dump(yaml_schema, Dumper=yaml.Dumper)

    @classmethod
    def singular(cls):
        """Return the 'singular' name of the CRD.

        Uses the `__singular__` dunder attribute if defined, otherwise
        defaults to the lowercase class name.
        """
        return getattr(cls, "__singular__", cls.__name__.lower())

    @classmethod
    def plural(cls):
        """Plural name of the CRD.

        Uses the `__plural__` dunder attribute if defined, otherwise
        appends 's' to the singular name.
        """
        return getattr(cls, "__plural__", f"{cls.singular()}s")

    @classmethod
    def kind(cls):
        """Return the kind of the CRD.

        Uses the `__kind__` dunder attribute if defined, otherwise
        defaults to the class name.
        """
        return getattr(cls, "__kind__", cls.__name__)

    @classmethod
    def crd_schema_dict(cls):
        """Return cls serialized as a Kubernetes CRD schema dict.

        This returns a dict representation of the Kubernetes CRD Object of cls,
        using class-level dunder attributes for full customization.
        """

        crd = {
            "apiVersion": "apiextensions.k8s.io/v1",
            "kind": "CustomResourceDefinition",
            "metadata": {
                "name": f"{cls.plural()}.{cls.__group__}",
            },
            "spec": {
                "group": cls.__group__,
                "scope": getattr(cls, "__scope__", "Namespaced"),
                "names": {
                    "singular": cls.singular(),
                    "plural": cls.plural(),
                    "kind": cls.kind(),
                    "shortNames": getattr(cls, "__shortnames__", []),
                },
                "versions": [
                    {
                        "name": cls.__version__,
                        # This API is served by default, currently there is no
                        # support for multiple versions.
                        "served": True,
                        "storage": True,
                        "schema": {
                            "openAPIV3Schema": {
                                **cls.apischema(),
                            }
                        },
                        "subresources": {
                            "status": {},
                        },
                    }
                ],
                # Kubernetes v1 CRDs require a conversion strategy. 'None' is the default
                # if only one version is served and stored, but explicitly setting it
                # is good practice. If multiple versions were supported, a webhook strategy
                # would likely be needed.
                "conversion": {"strategy": "None"},
            },
        }

        return crd

    @classmethod
    def crd_schema(cls):
        """Serialized YAML representation of Kubernetes CRD definition for cls.

        This serializes the dict representation from
        :py:method:`crd_schema_dict` to YAML.
        """
        # Directly dump the dictionary to YAML.
        return yaml.dump(
            yaml.load(json.dumps(cls.crd_schema_dict()), Loader=yaml.Loader),
            Dumper=yaml.Dumper,
        )

    @classmethod
    def _validate_api_version_and_kind(cls, json_data):
        expected_api_version = f"{cls.__group__}/{cls.__version__}"
        actual_api_version = json_data.get("apiVersion")
        actual_kind = json_data.get("kind")

        if actual_api_version != expected_api_version:
            raise ValueError(
                f"Invalid apiVersion: {actual_api_version} (expected {expected_api_version})"
            )

        if actual_kind != cls.kind():
            raise ValueError(f"Invalid kind: {actual_kind} (expected {cls.kind()})")

    @classmethod
    def _process_metadata(cls, metadata_data):
        inputs = {
            ObjectMeta_attribute_map.get(k, k): v for k, v in metadata_data.items()
        }
        return V1ObjectMeta(**inputs)

    @classmethod
    def _process_spec(cls, spec_data):
        spec_cls = get_attr_class(cls, "spec")
        if spec_cls is None:
            return spec_data
        return apischema_deserialize(
            spec_cls,
            spec_data,
            aliaser=to_camel_case if getattr(cls, "__camel_case__", True) else None,
        )

    @classmethod
    def _process_status(cls, status_data):
        status_cls = get_attr_class(cls, "status")
        if not status_cls:
            return status_data
        return apischema_deserialize(
            status_cls,
            status_data,
            aliaser=to_camel_case if getattr(cls, "__camel_case__", True) else None,
        )

    @classmethod
    def from_json(cls, json_data: dict):
        """Instantiate the class from JSON data fetched from Kubernetes.

        :param json_data: The CR JSON returned from Kubernetes API.
        :type json_data: Dict
        :returns: Instantiated cls with the data from json_data.
        :rtype: cls
        """
        cls._validate_api_version_and_kind(json_data)

        metadata = cls._process_metadata(json_data.get("metadata", {}))
        spec = cls._process_spec(json_data.get("spec", {}))

        ins = cls(spec=spec)
        ins.json = json_data
        ins.metadata = metadata

        if "status" in json_data:
            ins.status = cls._process_status(json_data["status"])

        return ins

    @classmethod
    def install(cls, k8s_client=None, exist_ok=True):
        """Install the CRD in Kubernetes.

        :param k8s_client: Instantiated Kubernetes API Client.
        :type k8s_client: kubernetes.client.api_client.ApiClient
        :param exist_ok: Boolean representing if error should be raised when
            trying to install a CRD that was already installed.
        :type exist_ok: bool
        """
        # Check Kubernetes version compatibility
        k8s_client = get_k8s_client(k8s_client)

        try:
            version_api = kubernetes.client.VersionApi(k8s_client)
            version_info = version_api.get_code()
            k8s_version = version_info.git_version

            version_match = re.search(r"v(\d+\.\d+)", k8s_version)
            if version_match:
                major_minor = float(version_match.group(1))
                if major_minor < 1.16:
                    logger.warning(
                        f"Kubernetes version {k8s_version} may not fully support "
                        f"apiextensions.k8s.io/v1 CRDs. Version 1.16+ is recommended."
                    )
            else:
                logger.warning(f"Could not parse Kubernetes version: {k8s_version}")
        except Exception as e:
            logger.warning(f"Failed to check Kubernetes version: {str(e)}")

        # Rest of the install method remains the same
        try:
            utils.create_from_yaml(
                k8s_client,
                yaml_objects=[yaml.load(cls.crd_schema(), Loader=yaml.Loader)],
            )
        except utils.FailToCreateError as e:
            code = json.loads(e.api_exceptions[0].body).get("code")
            if code == 409 and exist_ok:
                return
            raise

    @classmethod
    async def async_install(cls, k8s_client=None, exist_ok=True):
        """Asynchronously install the CRD in Kubernetes."""
        from kubernetes_asyncio.client import ApiextensionsV1Api, VersionApi
        from kubernetes_asyncio.client.rest import ApiException

        k8s_client = await get_k8s_async_client(k8s_client)
        # Check Kubernetes version compatibility
        try:
            version_api = VersionApi(k8s_client)
            version_info = await version_api.get_code()
            k8s_version = version_info.git_version

            version_match = re.search(r"v(\d+\.\d+)", k8s_version)
            if version_match:
                major_minor = float(version_match.group(1))
                if major_minor < 1.16:
                    logger.warning(
                        f"Kubernetes version {k8s_version} may not fully support "
                        f"apiextensions.k8s.io/v1 CRDs. Version 1.16+ is recommended."
                    )
            else:
                logger.warning(f"Could not parse Kubernetes version: {k8s_version}")
        except Exception as e:
            logger.warning(f"Failed to check Kubernetes version: {str(e)}")

        # Rest of the async install method
        api = ApiextensionsV1Api(k8s_client)
        crd_manifest = cls.crd_schema_dict()

        try:
            await api.create_custom_resource_definition(crd_manifest)
        except ApiException as e:
            if e.status == 409 and exist_ok:
                # CRD already exists
                return
            logger.error(f"Failed to create CRD: {e}", exc_info=True)
            raise

    @classmethod
    def watch(cls, k8s_client=None):
        """List and watch the changes in the Resource in Cluster."""
        k8s_client = get_k8s_client(k8s_client)
        api_instance = kubernetes.client.CustomObjectsApi(k8s_client)
        watch = kubernetes.watch.Watch()
        for event in watch.stream(
            func=api_instance.list_cluster_custom_object,
            group=cls.__group__,
            version=cls.__version__,
            plural=cls.plural().lower(),
            watch=True,
            allow_watch_bookmarks=True,
            timeout_seconds=50,
        ):
            obj = cls.from_json(event["object"])
            yield (event["type"], obj)

    @classmethod
    async def async_watch(cls, k8s_client=None):
        """Similar to watch, but uses async Kubernetes client for aio."""
        from kubernetes_asyncio import client, watch

        k8s_client = await get_k8s_async_client(k8s_client)
        api_instance = client.CustomObjectsApi(k8s_client)
        watch = watch.Watch()
        stream = watch.stream(
            func=api_instance.list_cluster_custom_object,
            group=cls.__group__,
            version=cls.__version__,
            plural=cls.plural().lower(),
            watch=True,
        )
        async for event in stream:
            obj = cls.from_json(event["object"])
            yield (event["type"], obj)

    def serialize(self, name=None, metadata=None):
        """Serialize the CR as a JSON suitable for POST'ing to K8s API.

        :param name: Optional name for the resource. If None, a name will be generated.
        :type name: str
        :param metadata: Optional dict of metadata to include (will be merged with default metadata).
        :type metadata: dict
        :returns: Dict representing the Kubernetes custom resource.
        :rtype: dict
        """
        use_camel = getattr(self, "__camel_case__", True)

        # Generate a default name if one isn't provided
        if name is None:
            name = self.generate_resource_name(prefix=self.singular())

        # Start with basic required metadata
        resource_metadata = {"name": name}

        # Merge any provided metadata
        if metadata and isinstance(metadata, dict):
            resource_metadata.update(metadata)

        return {
            "kind": self.kind(),
            "apiVersion": f"{self.__group__}/{self.__version__}",
            **apischema_serialize(self, aliaser=to_camel_case if use_camel else None),
            "metadata": resource_metadata,
        }

    def create(self, k8s_client=None, namespace="default", name=None, metadata=None):
        """Add the instance of this class as a K8s custom resource if it does not exists.
        To update an existing resource, use the `save` method.

        :param k8s_client: Kubernetes API client
        :param namespace: Namespace to create the resource in
        :param name: Optional name for the resource
        :param metadata: Optional additional metadata
        """
        k8s_client = get_k8s_client(k8s_client)
        api_instance = kubernetes.client.CustomObjectsApi(k8s_client)
        body_data = self.serialize(name=name, metadata=metadata)
        resp = api_instance.create_namespaced_custom_object(
            group=self.__group__,
            namespace=namespace,
            version=self.__version__,
            plural=self.plural(),
            body=body_data,
        )
        return resp

    async def async_create(
        self, k8s_client=None, namespace="default", name=None, metadata=None
    ):
        """Add the instance of this class as a K8s custom resource asynchronously."""
        from kubernetes_asyncio import client

        k8s_client = await get_k8s_async_client(k8s_client)
        api_instance = client.CustomObjectsApi(k8s_client)
        resp = await api_instance.create_namespaced_custom_object(
            group=self.__group__,
            namespace=namespace,
            version=self.__version__,
            plural=self.plural(),
            body=self.serialize(name=name, metadata=metadata),
        )
        return resp

    def save(
        self,
        k8s_client=None,
        namespace="default",
        name=None,
        metadata=None,
        field_manager="my-operator-default-fm",
        update_status=False,
    ):
        """Save the instance of this class as a K8s custom resource using Server-Side Apply.

        :param k8s_client: Kubernetes API client
        :param namespace: Namespace to create/update the resource in
        :param name: Optional name for the resource (if not set, taken from instance or metadata)
        :param metadata: Optional additional metadata to merge into the object's metadata
        :param field_manager: The name of the field manager for Server-Side Apply.
                              It's good practice to make this specific to your operator/controller.
        :param update_status: If True and status exists, also update the status subresource.
                              Note: Kubernetes ignores status in the main resource endpoint when
                              status is defined as a subresource, so this makes a separate API call.
        """

        k8s_client = get_k8s_client(k8s_client)  # Your helper to get a client
        k8s_client.set_default_header("Content-Type", "application/apply-patch+yaml")
        api_instance = kubernetes.client.CustomObjectsApi(k8s_client)

        # Serialize the full desired state.
        # The `name` parameter here helps ensure body_data['metadata']['name'] is set.
        body_data = self.serialize(name=name, metadata=metadata)

        # Ensure the name is set in the body_data
        if name and (
            "metadata" not in body_data or "name" not in body_data["metadata"]
        ):
            if "metadata" not in body_data:
                body_data["metadata"] = {}
            body_data["metadata"]["name"] = name

        resource_name = body_data.get("metadata", {}).get("name")
        if not resource_name:
            raise ValueError(
                "Resource name must be present in the serialized body's metadata for Server-Side Apply."
            )

        # For SSA, resourceVersion should NOT be in the body.
        if "metadata" in body_data and "resourceVersion" in body_data["metadata"]:
            del body_data["metadata"]["resourceVersion"]

        # For Server-Side Apply, we need to convert the body_data to YAML.
        # Python Obj -> JSON -> pure Python primitives -> YAML String
        body_data_obj = yaml.dump(
            yaml.load(json.dumps(body_data), Loader=yaml.Loader),
            Dumper=yaml.Dumper,
        )

        resp = api_instance.patch_namespaced_custom_object(
            group=self.__group__,
            version=self.__version__,
            namespace=namespace,
            plural=self.plural(),
            name=resource_name,  # Name from the body's metadata
            body=body_data_obj,
            field_manager=field_manager,
            force=True,
        )

        # If status exists and update_status is True, update the status subresource
        if update_status and hasattr(self, "status") and self.status is not None:
            self.update_status(
                k8s_client=k8s_client,
                name=resource_name,
                namespace=namespace,
            )

        return resp

    async def async_save(
        self,
        k8s_client=None,
        namespace="default",
        name=None,
        metadata=None,
        field_manager="my-operator-default-fm",
        update_status=False,
    ):
        """Save the instance of this class as a K8s custom resource asynchronously using Server-Side Apply.

        :param k8s_client: Kubernetes API client (async)
        :param namespace: Namespace to create/update the resource in
        :param name: Optional name for the resource (if not set, taken from instance or metadata)
        :param metadata: Optional additional metadata to merge into the object's metadata
        :param field_manager: The name of the field manager for Server-Side Apply.
        :param update_status: If True and status exists, also update the status subresource.
                              Note: Kubernetes ignores status in the main resource endpoint when
                              status is defined as a subresource, so this makes a separate API call.
        """
        from kubernetes_asyncio import (
            client as async_client,
        )

        k8s_client = await get_k8s_async_client(k8s_client)
        k8s_client.set_default_header("Content-Type", "application/apply-patch+yaml")

        api_instance = async_client.CustomObjectsApi(k8s_client)

        body_data = self.serialize(name=name, metadata=metadata)

        # Ensure the name is set in the body_data
        if name and (
            "metadata" not in body_data or "name" not in body_data["metadata"]
        ):
            if "metadata" not in body_data:
                body_data["metadata"] = {}
            body_data["metadata"]["name"] = name

        resource_name = body_data.get("metadata", {}).get("name")
        if not resource_name:
            raise ValueError(
                "Resource name must be present in the serialized body's metadata for Server-Side Apply."
            )

        if "metadata" in body_data and "resourceVersion" in body_data["metadata"]:
            del body_data["metadata"]["resourceVersion"]

        body_data_obj = yaml.dump(
            yaml.load(json.dumps(body_data), Loader=yaml.Loader),
            Dumper=yaml.Dumper,
        )

        resp = await api_instance.patch_namespaced_custom_object(
            group=self.__group__,
            version=self.__version__,
            namespace=namespace,
            plural=self.plural(),
            name=resource_name,
            body=body_data_obj,
            field_manager=field_manager,
            force=True,
        )

        # If status exists and update_status is True, update the status subresource
        if update_status and hasattr(self, "status") and self.status is not None:
            await self.async_update_status(
                k8s_client=k8s_client,
                name=resource_name,
                namespace=namespace,
            )

        return resp

    def generate_resource_name(self, prefix=None, include_hash=True):
        """Generate a deterministic name for this resource.

        :param prefix: Optional prefix for the name. Defaults to lowercase class name.
        :param include_hash: Whether to include a hash of object contents for uniqueness.
        :returns: A valid Kubernetes resource name.
        """
        if prefix is None:
            prefix = self.__class__.__name__.lower()

        # Ensure prefix is a valid DNS subdomain name
        prefix = re.sub(r"[^a-z0-9-]", "-", prefix.lower())
        prefix = re.sub(r"-+", "-", prefix)  # Replace multiple dashes with single dash
        prefix = prefix.strip("-")  # Remove leading/trailing dashes

        if not include_hash:
            return prefix

        # Generate a deterministic hash based on the serialized spec
        import hashlib

        use_camel = getattr(self, "__camel_case__", True)
        spec_json = json.dumps(
            apischema_serialize(self, aliaser=to_camel_case if use_camel else None),
            sort_keys=True,
        )
        hash_suffix = hashlib.md5(spec_json.encode()).hexdigest()[:8]

        # Combine prefix with hash, ensuring name isn't too long (k8s limit is 253 chars)
        max_prefix_len = 240 - len(hash_suffix) - 1  # leave room for dash and hash
        if len(prefix) > max_prefix_len:
            prefix = prefix[:max_prefix_len]

        return f"{prefix}-{hash_suffix}"

    def update_status(
        self, k8s_client=None, name=None, namespace="default", patch_operations=None
    ):
        """Update only the status subresource of this custom resource.

        :param k8s_client: Kubernetes API client
        :type k8s_client: kubernetes.client.api_client.ApiClient
        :param name: Optional name for the resource
        :type name: str
        :raises ValueError: If the resource name is not set in metadata
        :param namespace: Namespace where the resource exists
        :type namespace: str
        :param patch_operations: Optional patch operations to apply
            e.g., If current status is {"conditions": [{"type": "Ready", "status": "True"}], "message": "Old message"}
                patch_operations = [
                    {"op": "replace", "path": "/message", "value": "New critical message"},
                    {"op": "add", "path": "/conditions/-", "value": {"type": "Degraded", "status": "True"}} # Adds to end of conditions array
                ]
        :type patch_operations: dict
        :raises ValueError: If the status field does not exist
        """
        if not hasattr(self, "status"):
            raise ValueError(
                "Cannot update status: no status field exists on this object"
            )

        # Ensure the name is set in the object's metadata
        resource_name = self.ensure_resource_name(name=name)
        if not resource_name:
            raise ValueError(
                "Cannot update status: resource must have a name in metadata"
            )

        k8s_client = get_k8s_client(k8s_client)

        # If batch_operations is provided, we need to handle it differently
        if patch_operations:
            k8s_client.set_default_header("Content-Type", "application/json-patch+json")
            api_instance = kubernetes.client.CustomObjectsApi(k8s_client)
            api_instance.patch_namespaced_custom_object_status(
                group=self.__group__,
                version=self.__version__,
                namespace=namespace,
                plural=self.plural(),
                name=resource_name,
                body=patch_operations,
            )
            return

        api_instance = kubernetes.client.CustomObjectsApi(k8s_client)

        # Get status data, handling different status types
        if hasattr(self.status, "serialize"):
            status_data = self.status.serialize()
        elif hasattr(self.status, "__dict__"):
            use_camel = getattr(self, "__camel_case__", True)
            status_data = {
                to_camel_case(k) if use_camel else k: v
                for k, v in self.status.__dict__.items()
                if not k.startswith("_")  # Exclude private attributes
            }
        else:
            # Assume it's a dict or dict-like
            use_camel = getattr(self, "__camel_case__", True)
            status_data = {
                to_camel_case(k) if use_camel else k: v for k, v in self.status.items()
            }

        # Create the status patch
        status_obj = {"status": status_data}

        return api_instance.patch_namespaced_custom_object_status(
            group=self.__group__,
            version=self.__version__,
            namespace=namespace,
            plural=self.plural(),
            name=resource_name,
            body=status_obj,
        )

    async def async_update_status(
        self, k8s_client=None, name=None, namespace="default", patch_operations=None
    ):
        """Update only the status subresource of this custom resource asynchronously."""
        from kubernetes_asyncio import client

        if not hasattr(self, "status"):
            raise ValueError(
                "Cannot update status: no status field exists on this object"
            )

        # Ensure the name is set in the object's metadata
        resource_name = self.ensure_resource_name(name=name)
        if not resource_name:
            raise ValueError(
                "Cannot update status: resource must have a name in metadata"
            )

        k8s_client = await get_k8s_async_client(k8s_client)
        # If batch_operations is provided, we need to handle it differently
        if patch_operations:
            k8s_client.set_default_header("Content-Type", "application/json-patch+json")
            api_instance = client.CustomObjectsApi(k8s_client)
            await api_instance.patch_namespaced_custom_object_status(
                group=self.__group__,
                version=self.__version__,
                namespace=namespace,
                plural=self.plural(),
                name=resource_name,
                body=patch_operations,
            )
            return

        api_instance = client.CustomObjectsApi(k8s_client)
        # Get status data, handling different status types
        if hasattr(self.status, "serialize"):
            status_data = self.status.serialize()
        elif hasattr(self.status, "__dict__"):
            use_camel = getattr(self, "__camel_case__", True)
            status_data = {
                to_camel_case(k) if use_camel else k: v
                for k, v in self.status.__dict__.items()
                if not k.startswith("_")
            }
        else:
            # Assume it's a dict or dict-like
            use_camel = getattr(self, "__camel_case__", True)
            status_data = {
                to_camel_case(k) if use_camel else k: v for k, v in self.status.items()
            }

        # Create the status patch
        status_obj = {"status": status_data}

        return await api_instance.patch_namespaced_custom_object_status(
            group=self.__group__,
            version=self.__version__,
            namespace=namespace,
            plural=self.plural(),
            name=resource_name,
            body=status_obj,
        )

    def ensure_resource_name(self, name=None):
        """
        Ensures the resource has a name in its metadata.
        Returns the resource name.

        :param name: Optional name to set if metadata doesn't have one
        :return: The name of the resource
        :raises: ValueError if name cannot be determined or set
        """
        # Normalize metadata to a plain dict
        raw_meta = getattr(self, "metadata", None)
        is_dict_style = True
        if isinstance(raw_meta, V1ObjectMeta):
            meta_dict = raw_meta.to_dict()
            is_dict_style = False
        elif isinstance(raw_meta, dict):
            meta_dict = raw_meta.copy()
        else:
            meta_dict = {}

        # Inject the name if provided and missing
        if name:
            meta_dict.setdefault("name", name)

        # Validate
        resource_name = meta_dict.get("name")
        if not resource_name:
            return None

        # Write back either as dict or as V1ObjectMeta, depending on the API client
        self.metadata = meta_dict if is_dict_style else V1ObjectMeta(**meta_dict)

        return resource_name

    # Add these methods to the KubeResourceBase class

    @classmethod
    def get(cls, name, namespace="default", k8s_client=None):
        """
        Retrieve a specific instance of the custom resource.

        Args:
            name: Name of the resource to retrieve
            namespace: Kubernetes namespace (defaults to 'default')
            k8s_client: Instantiated Kubernetes API Client (optional)

        Returns:
            An instance of the class populated with the resource data
        """
        k8s_client = get_k8s_client(k8s_client)

        api_instance = k8s_sync_client.CustomObjectsApi(k8s_client)
        response = api_instance.get_namespaced_custom_object(
            group=cls.__group__,
            version=cls.__version__,
            namespace=namespace,
            plural=cls.plural(),
            name=name,
        )

        return cls.from_json(response)

    @classmethod
    async def async_get(cls, name, namespace="default", k8s_client=None):
        """
        Asynchronously retrieve a specific instance of the custom resource.

        Args:
            name: Name of the resource to retrieve
            namespace: Kubernetes namespace (defaults to 'default')
            k8s_client: Instantiated async Kubernetes API Client (optional)

        Returns:
            An instance of the class populated with the resource data
        """
        k8s_client = await get_k8s_async_client(k8s_client)

        api_instance = k8s_async_client.CustomObjectsApi(k8s_client)
        response = await api_instance.get_namespaced_custom_object(
            group=cls.__group__,
            version=cls.__version__,
            namespace=namespace,
            plural=cls.plural(),
            name=name,
        )

        return cls.from_json(response)

    @classmethod
    def list(cls, namespace="default", label_selector=None, k8s_client=None):
        """
        List all instances of the custom resource.

        Args:
            namespace: Kubernetes namespace (defaults to 'default')
            label_selector: Label selector string to filter resources
            k8s_client: Instantiated Kubernetes API Client (optional)

        Returns:
            A list of instances of the class
        """
        k8s_client = get_k8s_client(k8s_client)

        api_instance = k8s_sync_client.CustomObjectsApi(k8s_client)
        response = api_instance.list_namespaced_custom_object(
            group=cls.__group__,
            version=cls.__version__,
            namespace=namespace,
            plural=cls.plural(),
            label_selector=label_selector,
        )

        return [cls.from_json(item) for item in response.get("items", [])]

    def delete(self, namespace, name, k8s_client=None):
        """
        Delete a specific instance of the custom resource.

        Args:
            namespace: Kubernetes namespace
            name: Name of the resource to delete
            k8s_client: Instantiated Kubernetes API Client (optional)

        Returns:
            The API response from the delete operation
        """
        k8s_client = get_k8s_client(k8s_client)

        api_instance = k8s_sync_client.CustomObjectsApi(k8s_client)

        return api_instance.delete_namespaced_custom_object(
            group=self.__group__,
            version=self.__version__,
            namespace=namespace,
            plural=self.plural(),
            name=name,
        )

    async def async_delete(self, namespace, name, k8s_client=None):
        """
        Asynchronously delete a specific instance of the custom resource.

        Args:
            namespace: Kubernetes namespace
            name: Name of the resource to delete
            k8s_client: Instantiated async Kubernetes API Client (optional)

        Returns:
            The API response from the delete operation
        """
        k8s_client = await get_k8s_async_client(k8s_client)

        api_instance = k8s_async_client.CustomObjectsApi(k8s_client)

        return await api_instance.delete_namespaced_custom_object(
            group=self.__group__,
            version=self.__version__,
            namespace=namespace,
            plural=self.plural(),
            name=name,
        )
