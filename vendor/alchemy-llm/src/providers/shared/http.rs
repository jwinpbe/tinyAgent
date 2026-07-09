//! HTTP client utilities for providers.

use std::collections::HashMap;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

/// Build HTTP client with bearer auth and merged headers.
///
/// Delegates to [`build_headers`]; see it for the auth/header rules.
pub fn build_http_client(
    api_key: Option<&str>,
    model_headers: Option<&HashMap<String, String>>,
    extra_headers: Option<&HashMap<String, String>>,
) -> Result<reqwest::Client, crate::Error> {
    let headers = build_headers(api_key, model_headers, extra_headers)?;

    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(crate::Error::from)
}

/// Build the default header map for a request.
///
/// Always sets `Content-Type: application/json`. Adds `Authorization: Bearer
/// <api_key>` only when `api_key` is `Some`, then merges model and request
/// headers (model headers take precedence over the auth/content-type defaults,
/// request headers take precedence over model headers).
pub(crate) fn build_headers(
    api_key: Option<&str>,
    model_headers: Option<&HashMap<String, String>>,
    extra_headers: Option<&HashMap<String, String>>,
) -> Result<HeaderMap, crate::Error> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(api_key) = api_key {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| crate::Error::InvalidHeader(e.to_string()))?,
        );
    }

    merge_headers(&mut headers, model_headers);
    merge_headers(&mut headers, extra_headers);

    Ok(headers)
}

/// Merge optional headers into HeaderMap.
///
/// Invalid header names or values are silently skipped.
pub fn merge_headers(target: &mut HeaderMap, source: Option<&HashMap<String, String>>) {
    let Some(source) = source else { return };
    for (key, value) in source {
        if let (Ok(name), Ok(val)) = (
            HeaderName::try_from(key.as_str()),
            HeaderValue::from_str(value),
        ) {
            target.insert(name, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_headers_adds_valid() {
        let mut target = HeaderMap::new();
        let source = HashMap::from([
            ("X-Custom".to_string(), "value".to_string()),
            ("X-Another".to_string(), "test".to_string()),
        ]);
        merge_headers(&mut target, Some(&source));
        assert_eq!(target.len(), 2);
    }

    #[test]
    fn merge_headers_skips_invalid() {
        let mut target = HeaderMap::new();
        let source = HashMap::from([
            ("X-Valid".to_string(), "ok".to_string()),
            ("Invalid\nHeader".to_string(), "bad".to_string()),
        ]);
        merge_headers(&mut target, Some(&source));
        assert_eq!(target.len(), 1);
    }

    #[test]
    fn merge_headers_handles_none() {
        let mut target = HeaderMap::new();
        target.insert("X-Existing", HeaderValue::from_static("value"));
        merge_headers(&mut target, None);
        assert_eq!(target.len(), 1);
    }

    #[test]
    fn build_headers_includes_bearer_authorization_when_key_present() {
        let headers = build_headers(Some("secret-key"), None, None).expect("headers should build");

        assert_eq!(
            headers.get(AUTHORIZATION).unwrap(),
            HeaderValue::from_str("Bearer secret-key").unwrap()
        );
        assert_eq!(
            headers.get(CONTENT_TYPE).unwrap(),
            HeaderValue::from_static("application/json")
        );
    }

    #[test]
    fn build_headers_omits_authorization_when_no_key() {
        let headers = build_headers(None, None, None).expect("headers should build");

        assert!(headers.get(AUTHORIZATION).is_none());
        assert_eq!(
            headers.get(CONTENT_TYPE).unwrap(),
            HeaderValue::from_static("application/json")
        );
    }

    #[test]
    fn build_headers_merges_model_and_request_headers() {
        let model_headers = HashMap::from([("X-Model".to_string(), "model".to_string())]);
        let request_headers = HashMap::from([("X-Request".to_string(), "request".to_string())]);

        let headers = build_headers(Some("key"), Some(&model_headers), Some(&request_headers))
            .expect("headers should build");

        assert_eq!(
            headers.get("X-Model").unwrap(),
            HeaderValue::from_static("model")
        );
        assert_eq!(
            headers.get("X-Request").unwrap(),
            HeaderValue::from_static("request")
        );
    }
}
