import json
import re

import kubernetes
import yaml
from apischema import serialize
from apischema.json_schema import deserialization_schema
from kubernetes import utils
from kubernetes.client.models.v1_object_meta import V1ObjectMeta

# ObjectMeta_attribute_map is simply the reverse of the
# V1ObjectMeta.attribute_map , which is a mapping from python attribute to json
# key while this is the opposite from json key to python attribute so that we
# can pass in the values to instantiate the V1ObjectMeta object.
ObjectMeta_attribute_map = {
    value: key for key, value in V1ObjectMeta.attribute_map.items()
}


def to_camel_case(snake_str):
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # and join them all together.
    return components[0] + "".join(x.title() for x in components[1:])


def to_snake_case(camel_str):
    # Use regex to find uppercase letters that are not at the beginning of the string
    # and replace them with an underscore followed by the lowercase letter.
    # Also handles the case of consecutive uppercase letters.
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


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
        return yaml.dump(cls.apischema(), Dumper=yaml.Dumper)
        # yaml_schema = yaml.load(cls.apischema_json(), Loader=yaml.Loader)
        # return yaml.dump(yaml_schema, Dumper=yaml.Dumper)

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
                    "kind": cls.__name__,
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
                                "type": "object",
                                "properties": {
                                    "spec": cls.apischema(),
                                    # The status field's schema is provided by a class method
                                    # if available, otherwise it's an empty object schema.
                                    "status": getattr(
                                        cls,
                                        "status_schema",
                                        lambda: {
                                            "type": "object"
                                        },  # Default to object schema for status
                                    )(),
                                },
                                # Kubernetes recommends disabling additionalProperties for structural schemas
                                # unless you explicitly need to allow unknown fields.
                                "additionalProperties": False,
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
            cls.crd_schema_dict(),
            Dumper=yaml.Dumper,
        )

    @classmethod
    def from_json(cls, json_data: dict):
        """Instantiate the class from JSON data fetched from Kubernetes.

        :param json_data: The CR JSON returned from Kubernetes API.
        :type json_data: Dict
        :returns: Instantiated cls with the data from json_data.
        :rtype: cls
        """
        expected_api_version = f"{cls.__group__}/{cls.__version__}"
        actual_api_version = json_data.get("apiVersion")
        actual_kind = json_data.get("kind")

        if actual_api_version != expected_api_version:
            raise ValueError(
                f"Invalid apiVersion: {actual_api_version} (expected {expected_api_version})"
            )

        if actual_kind != cls.__name__:
            raise ValueError(f"Invalid kind: {actual_kind} (expected {cls.__name__})")

        metadata = json_data.get("metadata", {})
        inputs = {ObjectMeta_attribute_map.get(k, k): v for k, v in metadata.items()}
        meta = V1ObjectMeta(**inputs)

        # spec_data = json_data.get('spec', {})
        spec_data = {to_snake_case(k): v for k, v in json_data.get("spec", {}).items()}
        ins = cls(**spec_data)

        # Attach raw JSON and parsed metadata
        ins.json = json_data
        ins.metadata = meta

        # Optionally attach status if your class supports it
        if hasattr(ins, "status") and "status" in json_data:
            status_data = {
                to_snake_case(k): v for k, v in json_data.get("status", {}).items()
            }
            ins.status = status_data

        return ins

    @classmethod
    def install(cls, k8s_client, exist_ok=True):
        """Install the CRD in Kubernetes.

        :param k8s_client: Instantiated Kubernetes API Client.
        :type k8s_client: kubernetes.client.api_client.ApiClient
        :param exist_ok: Boolean representing if error should be raised when
            trying to install a CRD that was already installed.
        :type exist_ok: bool
        """
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
    async def async_install(cls, k8s_client, exist_ok=True):
        """Asynchronously install the CRD in Kubernetes.

        :param k8s_client: Instantiated Kubernetes async API Client.
        :type k8s_client: kubernetes_asyncio.client.api_client.ApiClient
        :param exist_ok: If True, don't raise an error if the CRD already exists.
        :type exist_ok: bool
        """
        from kubernetes_asyncio.client import ApiextensionsV1Api
        from kubernetes_asyncio.client.rest import ApiException

        api = ApiextensionsV1Api(k8s_client)

        crd_manifest = cls.crd_schema_dict()

        try:
            await api.create_custom_resource_definition(crd_manifest)
        except ApiException as e:
            if e.status == 409 and exist_ok:
                # CRD already exists
                return
            raise

    @classmethod
    def watch(cls, client):
        """List and watch the changes in the Resource in Cluster."""
        api_instance = kubernetes.client.CustomObjectsApi(client)
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
    async def async_watch(cls, k8s_client):
        """Similar to watch, but uses async Kubernetes client for aio."""
        from kubernetes_asyncio import client, watch

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

    def serialize(self, name_prefix=None):
        """Serialize the CR as a JSON suitable for POST'ing to K8s API."""
        use_camel = getattr(self, "__camel_case__", True)
        if name_prefix is None:
            name_prefix = self.__class__.__name__.lower()

        return {
            "kind": self.__class__.__name__,
            "apiVersion": f"{self.__group__}/{self.__version__}",
            "spec": serialize(self, aliaser=to_camel_case if use_camel else None),
            "metadata": {
                "name": (name_prefix + str(id(self))).lower(),
            },
        }

    def save(self, k8s_client, namespace="default"):
        """Save the instance of this class as a K8s custom resource."""
        api_instance = kubernetes.client.CustomObjectsApi(k8s_client)
        resp = api_instance.create_namespaced_custom_object(
            group=self.__group__,
            namespace=namespace,
            version=self.__version__,
            plural=self.plural(),
            body=self.serialize(),
        )
        return resp

    async def async_save(self, k8s_client, namespace="default"):
        """Save the instance of this class as a K8s custom resource."""
        from kubernetes_asyncio import client

        api_instance = client.CustomObjectsApi(k8s_client)
        resp = await api_instance.create_namespaced_custom_object(
            group=self.__group__,
            namespace=namespace,
            version=self.__version__,
            plural=self.plural(),
            body=self.serialize(),
        )
        return resp
