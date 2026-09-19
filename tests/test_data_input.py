from frontend import data_input


def test_append_manual_record_to_payload():
    payload = {
        "shipments": [{"shipment_id": "S1"}],
        "vehicles": [],
        "hubs": [],
        "routes": [],
    }
    record = {
        "shipment_id": "S2",
        "origin": "Bengaluru",
        "current_location": "Bengaluru",
        "destination": "Chennai",
        "weight_kg": 100.0,
        "volume_m3": 1.0,
        "priority": "Critical",
        "deadline": "2026-09-20T00:00:00",
        "status": "Misplaced",
        "shipment_value": 50000.0,
        "created_at": "2026-09-19T00:00:00",
        "expected_transport_hours": 8.0,
        "delay_probability": 0.2,
        "route_risk": 0.2,
        "business_category": "Medicine",
    }

    result = data_input.local_append_record(payload, "shipments", record)

    assert len(result["shipments"]) == 2
    assert result["shipments"][-1]["shipment_id"] == "S2"


def test_append_csv_records_to_payload():
    payload = {"shipments": [], "vehicles": [], "hubs": [], "routes": []}
    csv_text = (
        "shipment_id,origin,current_location,destination,weight_kg,volume_m3,priority,deadline,status,shipment_value,"
        "created_at,expected_transport_hours,delay_probability,route_risk,business_category\n"
        "S3,Bengaluru,Bengaluru,Chennai,150,1.5,Critical,2026-09-20T00:00:00,Misplaced,70000,2026-09-19T00:00:00,8,0.2,0.2,Medicine"
    )

    result = data_input.local_append_csv(payload, "shipments", csv_text)

    assert len(result["shipments"]) == 1
    assert result["shipments"][0]["shipment_id"] == "S3"
