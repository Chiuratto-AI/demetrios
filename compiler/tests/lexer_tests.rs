//! Lexer tests

use demetrios::lexer::{lex, TokenKind};

#[test]
fn test_lex_empty() {
    let tokens = lex("").unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].kind, TokenKind::Eof);
}

#[test]
fn test_lex_whitespace() {
    let tokens = lex("   \t\n  ").unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].kind, TokenKind::Eof);
}

#[test]
fn test_lex_simple_let() {
    let tokens = lex("let x = 42").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Let);
    assert_eq!(tokens[1].kind, TokenKind::Ident);
    assert_eq!(tokens[1].text, "x");
    assert_eq!(tokens[2].kind, TokenKind::Eq);
    assert_eq!(tokens[3].kind, TokenKind::IntLit);
    assert_eq!(tokens[3].text, "42");
}

#[test]
fn test_lex_keywords() {
    let source = "fn let mut if else match for while loop return";
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::Fn);
    assert_eq!(tokens[1].kind, TokenKind::Let);
    assert_eq!(tokens[2].kind, TokenKind::Mut);
    assert_eq!(tokens[3].kind, TokenKind::If);
    assert_eq!(tokens[4].kind, TokenKind::Else);
    assert_eq!(tokens[5].kind, TokenKind::Match);
    assert_eq!(tokens[6].kind, TokenKind::For);
    assert_eq!(tokens[7].kind, TokenKind::While);
    assert_eq!(tokens[8].kind, TokenKind::Loop);
    assert_eq!(tokens[9].kind, TokenKind::Return);
}

#[test]
fn test_lex_d_keywords() {
    let source = "effect handler with perform linear affine kernel";
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::Effect);
    assert_eq!(tokens[1].kind, TokenKind::Handler);
    assert_eq!(tokens[2].kind, TokenKind::With);
    assert_eq!(tokens[3].kind, TokenKind::Perform);
    assert_eq!(tokens[4].kind, TokenKind::Linear);
    assert_eq!(tokens[5].kind, TokenKind::Affine);
    assert_eq!(tokens[6].kind, TokenKind::Kernel);
}

#[test]
fn test_lex_operators() {
    let source = "+ - * / % == != < <= > >= && || -> =>";
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::Plus);
    assert_eq!(tokens[1].kind, TokenKind::Minus);
    assert_eq!(tokens[2].kind, TokenKind::Star);
    assert_eq!(tokens[3].kind, TokenKind::Slash);
    assert_eq!(tokens[4].kind, TokenKind::Percent);
    assert_eq!(tokens[5].kind, TokenKind::EqEq);
    assert_eq!(tokens[6].kind, TokenKind::Ne);
    assert_eq!(tokens[7].kind, TokenKind::Lt);
    assert_eq!(tokens[8].kind, TokenKind::Le);
    assert_eq!(tokens[9].kind, TokenKind::Gt);
    assert_eq!(tokens[10].kind, TokenKind::Ge);
    assert_eq!(tokens[11].kind, TokenKind::AmpAmp);
    assert_eq!(tokens[12].kind, TokenKind::PipePipe);
    assert_eq!(tokens[13].kind, TokenKind::Arrow);
    assert_eq!(tokens[14].kind, TokenKind::FatArrow);
}

#[test]
fn test_lex_literals() {
    let source = r#"42 3.14 "hello" 'c' true false"#;
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::IntLit);
    assert_eq!(tokens[0].text, "42");

    assert_eq!(tokens[1].kind, TokenKind::FloatLit);
    assert_eq!(tokens[1].text, "3.14");

    assert_eq!(tokens[2].kind, TokenKind::StringLit);
    assert_eq!(tokens[2].text, "\"hello\"");

    assert_eq!(tokens[3].kind, TokenKind::CharLit);
    assert_eq!(tokens[3].text, "'c'");

    assert_eq!(tokens[4].kind, TokenKind::True);
    assert_eq!(tokens[5].kind, TokenKind::False);
}

#[test]
fn test_lex_delimiters() {
    let source = "( ) [ ] { } , ; : :: .";
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::LParen);
    assert_eq!(tokens[1].kind, TokenKind::RParen);
    assert_eq!(tokens[2].kind, TokenKind::LBracket);
    assert_eq!(tokens[3].kind, TokenKind::RBracket);
    assert_eq!(tokens[4].kind, TokenKind::LBrace);
    assert_eq!(tokens[5].kind, TokenKind::RBrace);
    assert_eq!(tokens[6].kind, TokenKind::Comma);
    assert_eq!(tokens[7].kind, TokenKind::Semi);
    assert_eq!(tokens[8].kind, TokenKind::Colon);
    assert_eq!(tokens[9].kind, TokenKind::ColonColon);
    assert_eq!(tokens[10].kind, TokenKind::Dot);
}

#[test]
fn test_lex_line_comment() {
    let source = "let x = 1 // this is a comment\nlet y = 2";
    let tokens = lex(source).unwrap();

    // Comments should be skipped
    let let_count = tokens.iter().filter(|t| t.kind == TokenKind::Let).count();
    assert_eq!(let_count, 2);
}

#[test]
fn test_lex_block_comment() {
    let source = "let /* comment */ x = 1";
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::Let);
    assert_eq!(tokens[1].kind, TokenKind::Ident);
    assert_eq!(tokens[1].text, "x");
}

#[test]
fn test_lex_function() {
    let source = "fn add(a: i32, b: i32) -> i32 { a + b }";
    let tokens = lex(source).unwrap();

    assert_eq!(tokens[0].kind, TokenKind::Fn);
    assert_eq!(tokens[1].kind, TokenKind::Ident);
    assert_eq!(tokens[1].text, "add");
    assert_eq!(tokens[2].kind, TokenKind::LParen);
    // ... etc
}

#[test]
fn test_lex_effect_signature() {
    let source = "fn greet(name: String) with IO { }";
    let tokens = lex(source).unwrap();

    let with_pos = tokens.iter().position(|t| t.kind == TokenKind::With);
    assert!(with_pos.is_some());
}
