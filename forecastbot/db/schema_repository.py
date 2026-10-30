from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import text

from forecastbot.db.postgres import get_forecastbot_engine


class SchemaKnowledgeRepository:
    def get_by_dataset(self, dataset_code: str) -> List[Dict[str, Any]]:
        engine = get_forecastbot_engine()

        sql = text("""
            SELECT
                dataset_code,
                dataset_name,
                dataset_group,
                domain_name,
                table_name,
                table_type,
                description,
                primary_date_column,
                primary_value_column,
                default_granularity,
                dimensions_json,
                measures_json,
                joins_json,
                allowed_filters_json,
                sample_queries_json,
                llm_schema_context,
                sql_generation_rules,
                canonical_ds_expression,
                canonical_y_expression,
                filterable_dimensions_json,
                examples_json
            FROM schema_knowledge
            WHERE dataset_code = :dataset_code
              AND is_active = TRUE
            ORDER BY
                CASE WHEN table_type = 'fact' THEN 1 ELSE 2 END,
                table_name
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, {"dataset_code": dataset_code}).mappings().all()

        return [dict(r) for r in rows]

    def get_by_dataset_code(self, dataset_code: str) -> List[Dict[str, Any]]:
        return self.get_by_dataset(dataset_code)

    def get_schema_knowledge_by_dataset(self, dataset_code: str) -> List[Dict[str, Any]]:
        return self.get_by_dataset(dataset_code)

    def list_by_dataset_code(self, dataset_code: str) -> List[Dict[str, Any]]:
        return self.get_by_dataset(dataset_code)