{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if docstring %}
{{ docstring }}
{% endif %}

{% if attributes %}
Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description
{% for name in attributes %}
   * - :attr:`~{{ module }}.{{ objname }}.{{ name }}`
     - {{ get_summary(module, objname, name) }}
{% endfor %}
{% endif %}

{% if methods %}
Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description
{% for name in methods %}
   * - :meth:`~{{ module }}.{{ objname }}.{{ name }}`
     - {{ get_summary(module, objname, name) }}
{% endfor %}
{% endif %}

{% if attributes %}
Property Details
----------------

{% for name in attributes %}
.. autoattribute:: {{ objname }}.{{ name }}

{% endfor %}
{% endif %}

{% if methods %}
Method Details
--------------

{% for name in methods %}
.. automethod:: {{ objname }}.{{ name }}

{% endfor %}
{% endif %}