using System;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Globalization;

public class FloatJsonConverter : JsonConverter<float>
{
    public override float Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.String)
        {
            // GetString()이 null이면 JsonException을 던져서 null 경고 해결
            string? stringValue = reader.GetString();
            if (stringValue == null) throw new JsonException("Expected string value.");

            return float.Parse(stringValue, CultureInfo.InvariantCulture);
        }
        return (float)reader.GetDouble();
    }

    public override void Write(Utf8JsonWriter writer, float value, JsonSerializerOptions options)
    {
        if (Math.Abs(value - Math.Truncate(value)) < float.Epsilon)
        {
            writer.WriteRawValue(value.ToString("0.0", CultureInfo.InvariantCulture));
        }
        else
        {
            writer.WriteNumberValue(value);
        }
    }
}

public class DoubleJsonConverter : JsonConverter<double>
{
    public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.String)
        {
            string? stringValue = reader.GetString();
            if (stringValue == null) throw new JsonException("Expected string value.");

            return double.Parse(stringValue, CultureInfo.InvariantCulture);
        }
        return reader.GetDouble();
    }

    public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
    {
        if (Math.Abs(value - Math.Truncate(value)) < double.Epsilon)
        {
            writer.WriteRawValue(value.ToString("0.0", CultureInfo.InvariantCulture));
        }
        else
        {
            writer.WriteNumberValue(value);
        }
    }
}
