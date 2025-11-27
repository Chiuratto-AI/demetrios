//! Parser for the Demetrios language
//!
//! A recursive descent parser that produces an AST from a token stream.

use crate::ast::*;
use crate::common::{IdGenerator, NodeId, Span};
use crate::lexer::{Token, TokenKind};
use miette::Result;

/// Parse a token stream into an AST
pub fn parse(tokens: &[Token], _source: &str) -> Result<Ast> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

/// Parser state
struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    id_gen: IdGenerator,
    /// When false, don't parse `Ident { ... }` as a struct literal
    /// This is needed to resolve ambiguity in contexts like `match x { ... }`
    allow_struct_literals: bool,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            id_gen: IdGenerator::new(),
            allow_struct_literals: true,
        }
    }

    fn next_id(&mut self) -> NodeId {
        self.id_gen.next()
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or_else(|| {
            self.tokens
                .last()
                .expect("token stream should have at least EOF")
        })
    }

    fn peek(&self) -> TokenKind {
        self.current().kind
    }

    fn peek_n(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos + n)
            .map(|t| t.kind)
            .unwrap_or(TokenKind::Eof)
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek() == kind
    }

    fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.contains(&self.peek())
    }

    fn advance(&mut self) -> &Token {
        let tok = self.current();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        // Return the token that was at the previous position
        &self.tokens[self.pos.saturating_sub(1)]
    }

    fn expect(&mut self, kind: TokenKind) -> Result<&Token> {
        if self.at(kind) {
            Ok(self.advance())
        } else {
            Err(miette::miette!(
                "Expected {:?}, found {:?} at position {}",
                kind,
                self.peek(),
                self.current().span.start
            ))
        }
    }

    fn span(&self) -> Span {
        self.current().span
    }

    // ==================== PROGRAM ====================

    fn parse_program(&mut self) -> Result<Ast> {
        let mut items = Vec::new();

        // Optional module declaration
        let module_name = if self.at(TokenKind::Module) {
            self.advance();
            let name = self.parse_path()?;
            Some(name)
        } else {
            None
        };

        // Parse items
        while !self.at(TokenKind::Eof) {
            items.push(self.parse_item()?);
        }

        Ok(Ast { module_name, items })
    }

    // ==================== ITEMS ====================

    fn parse_item(&mut self) -> Result<Item> {
        // Parse visibility
        let visibility = self.parse_visibility();

        // Parse modifiers
        let modifiers = self.parse_modifiers();

        match self.peek() {
            TokenKind::Fn | TokenKind::Kernel => self.parse_fn(visibility, modifiers),
            TokenKind::Let | TokenKind::Const => self.parse_global(visibility, modifiers),
            TokenKind::Struct => self.parse_struct(visibility, modifiers),
            TokenKind::Enum => self.parse_enum(visibility, modifiers),
            TokenKind::Trait => self.parse_trait(visibility, modifiers),
            TokenKind::Impl => self.parse_impl(),
            TokenKind::Type => self.parse_type_alias(visibility),
            TokenKind::Effect => self.parse_effect(visibility),
            TokenKind::Handler => self.parse_handler(visibility),
            TokenKind::Import => self.parse_import(),
            TokenKind::Extern => self.parse_extern(),
            _ => Err(miette::miette!(
                "Unexpected token {:?} at start of item",
                self.peek()
            )),
        }
    }

    fn parse_visibility(&mut self) -> Visibility {
        if self.at(TokenKind::Pub) {
            self.advance();
            Visibility::Public
        } else {
            Visibility::Private
        }
    }

    fn parse_modifiers(&mut self) -> Modifiers {
        let mut mods = Modifiers::default();

        loop {
            match self.peek() {
                TokenKind::Linear => {
                    self.advance();
                    mods.linear = true;
                }
                TokenKind::Affine => {
                    self.advance();
                    mods.affine = true;
                }
                TokenKind::Async => {
                    self.advance();
                    mods.is_async = true;
                }
                TokenKind::Unsafe => {
                    self.advance();
                    mods.is_unsafe = true;
                }
                _ => break,
            }
        }

        mods
    }

    // ==================== FUNCTIONS ====================

    fn parse_fn(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();

        let is_kernel = if self.at(TokenKind::Kernel) {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Fn)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        let effects = self.parse_effect_clause()?;
        let where_clause = self.parse_where_clause()?;
        let body = self.parse_block()?;

        let end = self.span();

        Ok(Item::Function(FnDef {
            id: self.next_id(),
            visibility,
            modifiers: FnModifiers {
                is_async: modifiers.is_async,
                is_unsafe: modifiers.is_unsafe,
                is_kernel,
            },
            name,
            generics,
            params,
            return_type,
            effects,
            where_clause,
            body,
            span: start.merge(end),
        }))
    }

    fn parse_params(&mut self) -> Result<Vec<Param>> {
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();

        while !self.at(TokenKind::RParen) {
            params.push(self.parse_param()?);
            if !self.at(TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param> {
        let is_mut = if self.at(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };

        // Handle special `self` parameter which doesn't require a type annotation
        if self.at(TokenKind::SelfLower) {
            self.advance();
            return Ok(Param {
                id: self.next_id(),
                is_mut,
                pattern: Pattern::Binding {
                    name: "self".to_string(),
                    mutable: false,
                },
                ty: TypeExpr::SelfType, // Special self type
            });
        }

        // Handle &self and &mut self
        if self.at(TokenKind::Amp) {
            self.advance();
            let is_ref_mut = if self.at(TokenKind::Mut) {
                self.advance();
                true
            } else {
                false
            };
            if self.at(TokenKind::SelfLower) {
                self.advance();
                return Ok(Param {
                    id: self.next_id(),
                    is_mut: is_ref_mut,
                    pattern: Pattern::Binding {
                        name: "self".to_string(),
                        mutable: is_ref_mut,
                    },
                    ty: TypeExpr::Reference {
                        mutable: is_ref_mut,
                        inner: Box::new(TypeExpr::SelfType),
                    },
                });
            }
            // Not &self, backtrack is tricky - for now just error
            return Err(miette::miette!("Expected 'self' after '&' in parameter"));
        }

        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(Param {
            id: self.next_id(),
            is_mut,
            pattern,
            ty,
        })
    }

    fn parse_return_type(&mut self) -> Result<Option<TypeExpr>> {
        if self.at(TokenKind::Arrow) {
            self.advance();
            Ok(Some(self.parse_type()?))
        } else {
            Ok(None)
        }
    }

    fn parse_effect_clause(&mut self) -> Result<Vec<EffectRef>> {
        if self.at(TokenKind::With) {
            self.advance();
            let mut effects = vec![self.parse_effect_ref()?];
            while self.at(TokenKind::Comma) {
                self.advance();
                effects.push(self.parse_effect_ref()?);
            }
            Ok(effects)
        } else {
            Ok(Vec::new())
        }
    }

    fn parse_effect_ref(&mut self) -> Result<EffectRef> {
        let name = self.parse_path()?;
        let args = if self.at(TokenKind::Lt) {
            self.parse_type_args()?
        } else {
            Vec::new()
        };
        Ok(EffectRef {
            id: self.next_id(),
            name,
            args,
        })
    }

    // ==================== STRUCTS ====================

    fn parse_struct(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Struct)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) {
            fields.push(self.parse_field()?);
            if !self.at(TokenKind::RBrace) {
                // Allow optional comma
                if self.at(TokenKind::Comma) {
                    self.advance();
                }
            }
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Struct(StructDef {
            id: self.next_id(),
            visibility,
            modifiers: TypeModifiers {
                linear: modifiers.linear,
                affine: modifiers.affine,
            },
            name,
            generics,
            where_clause,
            fields,
            span: start.merge(end),
        }))
    }

    fn parse_field(&mut self) -> Result<FieldDef> {
        let visibility = self.parse_visibility();
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(FieldDef {
            id: self.next_id(),
            visibility,
            name,
            ty,
        })
    }

    // ==================== ENUMS ====================

    fn parse_enum(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Enum)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) {
            variants.push(self.parse_variant()?);
            if !self.at(TokenKind::RBrace) {
                if self.at(TokenKind::Comma) {
                    self.advance();
                }
            }
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Enum(EnumDef {
            id: self.next_id(),
            visibility,
            modifiers: TypeModifiers {
                linear: modifiers.linear,
                affine: modifiers.affine,
            },
            name,
            generics,
            where_clause,
            variants,
            span: start.merge(end),
        }))
    }

    fn parse_variant(&mut self) -> Result<VariantDef> {
        let name = self.parse_ident()?;
        let data = if self.at(TokenKind::LParen) {
            self.advance();
            let mut types = Vec::new();
            while !self.at(TokenKind::RParen) {
                types.push(self.parse_type()?);
                if !self.at(TokenKind::RParen) {
                    self.expect(TokenKind::Comma)?;
                }
            }
            self.expect(TokenKind::RParen)?;
            VariantData::Tuple(types)
        } else if self.at(TokenKind::LBrace) {
            self.advance();
            let mut fields = Vec::new();
            while !self.at(TokenKind::RBrace) {
                fields.push(self.parse_field()?);
                if !self.at(TokenKind::RBrace) {
                    if self.at(TokenKind::Comma) {
                        self.advance();
                    }
                }
            }
            self.expect(TokenKind::RBrace)?;
            VariantData::Struct(fields)
        } else {
            VariantData::Unit
        };

        Ok(VariantDef {
            id: self.next_id(),
            name,
            data,
        })
    }

    // ==================== TRAITS & IMPL ====================

    fn parse_trait(&mut self, visibility: Visibility, _modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Trait)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let supertraits = if self.at(TokenKind::Colon) {
            self.advance();
            let mut traits = vec![self.parse_path()?];
            while self.at(TokenKind::Plus) {
                self.advance();
                traits.push(self.parse_path()?);
            }
            traits
        } else {
            Vec::new()
        };
        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) {
            items.push(self.parse_trait_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Trait(TraitDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            supertraits,
            where_clause,
            items,
            span: start.merge(end),
        }))
    }

    fn parse_trait_item(&mut self) -> Result<TraitItem> {
        let visibility = self.parse_visibility();
        let modifiers = self.parse_modifiers();

        match self.peek() {
            TokenKind::Fn => {
                self.advance();
                let name = self.parse_ident()?;
                let generics = self.parse_generics()?;
                let params = self.parse_params()?;
                let return_type = self.parse_return_type()?;
                let effects = self.parse_effect_clause()?;
                let where_clause = self.parse_where_clause()?;

                let default_body = if self.at(TokenKind::LBrace) {
                    Some(self.parse_block()?)
                } else {
                    self.expect(TokenKind::Semi)?;
                    None
                };

                Ok(TraitItem::Fn(TraitFnDef {
                    id: self.next_id(),
                    name,
                    generics,
                    params,
                    return_type,
                    effects,
                    where_clause,
                    default_body,
                }))
            }
            TokenKind::Type => {
                self.advance();
                let name = self.parse_ident()?;
                let bounds = if self.at(TokenKind::Colon) {
                    self.advance();
                    let mut bounds = vec![self.parse_path()?];
                    while self.at(TokenKind::Plus) {
                        self.advance();
                        bounds.push(self.parse_path()?);
                    }
                    bounds
                } else {
                    Vec::new()
                };
                let default = if self.at(TokenKind::Eq) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(TokenKind::Semi)?;

                Ok(TraitItem::Type(TraitTypeDef {
                    id: self.next_id(),
                    name,
                    bounds,
                    default,
                }))
            }
            _ => Err(miette::miette!(
                "Expected trait item, found {:?}",
                self.peek()
            )),
        }
    }

    fn parse_impl(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Impl)?;

        let generics = self.parse_generics()?;

        // Check if this is a trait impl
        let (trait_ref, target_type) = if self.peek_n(1) == TokenKind::For {
            let trait_path = self.parse_path()?;
            self.expect(TokenKind::For)?;
            let ty = self.parse_type()?;
            (Some(trait_path), ty)
        } else {
            (None, self.parse_type()?)
        };

        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) {
            items.push(self.parse_impl_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Impl(ImplDef {
            id: self.next_id(),
            generics,
            trait_ref,
            target_type,
            where_clause,
            items,
            span: start.merge(end),
        }))
    }

    fn parse_impl_item(&mut self) -> Result<ImplItem> {
        let visibility = self.parse_visibility();
        let modifiers = self.parse_modifiers();

        match self.peek() {
            TokenKind::Fn | TokenKind::Kernel => {
                let item = self.parse_fn(visibility, modifiers)?;
                if let Item::Function(f) = item {
                    Ok(ImplItem::Fn(f))
                } else {
                    unreachable!()
                }
            }
            TokenKind::Type => {
                self.advance();
                let name = self.parse_ident()?;
                self.expect(TokenKind::Eq)?;
                let ty = self.parse_type()?;
                self.expect(TokenKind::Semi)?;
                Ok(ImplItem::Type(ImplTypeDef {
                    id: self.next_id(),
                    name,
                    ty,
                }))
            }
            _ => Err(miette::miette!(
                "Expected impl item, found {:?}",
                self.peek()
            )),
        }
    }

    // ==================== TYPE ALIASES ====================

    fn parse_type_alias(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Type)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        self.expect(TokenKind::Eq)?;
        let ty = self.parse_type()?;
        self.expect(TokenKind::Semi)?;

        let end = self.span();

        Ok(Item::TypeAlias(TypeAliasDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            ty,
            span: start.merge(end),
        }))
    }

    // ==================== EFFECTS ====================

    fn parse_effect(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Effect)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;

        self.expect(TokenKind::LBrace)?;
        let mut operations = Vec::new();
        while !self.at(TokenKind::RBrace) {
            operations.push(self.parse_effect_op()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Effect(EffectDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            operations,
            span: start.merge(end),
        }))
    }

    fn parse_effect_op(&mut self) -> Result<EffectOpDef> {
        self.expect(TokenKind::Fn)?;
        let name = self.parse_ident()?;
        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        self.expect(TokenKind::Semi)?;

        Ok(EffectOpDef {
            id: self.next_id(),
            name,
            params,
            return_type,
        })
    }

    fn parse_handler(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Handler)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        self.expect(TokenKind::For)?;
        let effect = self.parse_path()?;

        self.expect(TokenKind::LBrace)?;
        let mut cases = Vec::new();
        while !self.at(TokenKind::RBrace) {
            cases.push(self.parse_handler_case()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Handler(HandlerDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            effect,
            cases,
            span: start.merge(end),
        }))
    }

    fn parse_handler_case(&mut self) -> Result<HandlerCase> {
        let name = self.parse_ident()?;
        let params = self.parse_params()?;
        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;
        if self.at(TokenKind::Comma) {
            self.advance();
        }

        Ok(HandlerCase {
            id: self.next_id(),
            name,
            params,
            body,
        })
    }

    // ==================== IMPORTS & EXTERN ====================

    fn parse_import(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Import)?;
        let path = self.parse_path()?;
        self.expect(TokenKind::Semi)?;
        let end = self.span();

        Ok(Item::Import(ImportDef {
            id: self.next_id(),
            path,
            span: start.merge(end),
        }))
    }

    fn parse_extern(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Extern)?;

        let abi = if self.at(TokenKind::StringLit) {
            let s = self.advance().text.clone();
            // Remove quotes
            s[1..s.len() - 1].to_string()
        } else {
            "C".to_string()
        };

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) {
            items.push(self.parse_extern_fn()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Extern(ExternBlock {
            id: self.next_id(),
            abi,
            items,
            span: start.merge(end),
        }))
    }

    fn parse_extern_fn(&mut self) -> Result<ExternFn> {
        self.expect(TokenKind::Fn)?;
        let name = self.parse_ident()?;
        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        self.expect(TokenKind::Semi)?;

        Ok(ExternFn {
            id: self.next_id(),
            name,
            params,
            return_type,
        })
    }

    // ==================== GLOBALS ====================

    fn parse_global(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        let is_const = self.at(TokenKind::Const);
        self.advance(); // let or const

        let is_mut = if self.at(TokenKind::Mut) && !is_const {
            self.advance();
            true
        } else {
            false
        };

        let pattern = self.parse_pattern()?;
        let ty = if self.at(TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;

        let end = self.span();

        Ok(Item::Global(GlobalDef {
            id: self.next_id(),
            visibility,
            is_const,
            is_mut,
            pattern,
            ty,
            value,
            span: start.merge(end),
        }))
    }

    // ==================== GENERICS ====================

    fn parse_generics(&mut self) -> Result<Generics> {
        if !self.at(TokenKind::Lt) {
            return Ok(Generics { params: Vec::new() });
        }

        self.advance();
        let mut params = Vec::new();

        while !self.at(TokenKind::Gt) {
            params.push(self.parse_generic_param()?);
            if !self.at(TokenKind::Gt) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::Gt)?;

        Ok(Generics { params })
    }

    fn parse_generic_param(&mut self) -> Result<GenericParam> {
        // Check for const generic
        if self.at(TokenKind::Const) {
            self.advance();
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            return Ok(GenericParam::Const { name, ty });
        }

        // Type parameter
        let name = self.parse_ident()?;
        let bounds = if self.at(TokenKind::Colon) {
            self.advance();
            let mut bounds = vec![self.parse_path()?];
            while self.at(TokenKind::Plus) {
                self.advance();
                bounds.push(self.parse_path()?);
            }
            bounds
        } else {
            Vec::new()
        };
        let default = if self.at(TokenKind::Eq) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        Ok(GenericParam::Type {
            name,
            bounds,
            default,
        })
    }

    fn parse_type_args(&mut self) -> Result<Vec<TypeExpr>> {
        self.expect(TokenKind::Lt)?;
        let mut args = Vec::new();

        while !self.at(TokenKind::Gt) {
            args.push(self.parse_type()?);
            if !self.at(TokenKind::Gt) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::Gt)?;
        Ok(args)
    }

    fn parse_where_clause(&mut self) -> Result<Vec<WherePredicate>> {
        if !self.at(TokenKind::Where) {
            return Ok(Vec::new());
        }

        self.advance();
        let mut predicates = Vec::new();

        loop {
            let ty = self.parse_type()?;
            self.expect(TokenKind::Colon)?;
            let mut bounds = vec![self.parse_path()?];
            while self.at(TokenKind::Plus) {
                self.advance();
                bounds.push(self.parse_path()?);
            }
            predicates.push(WherePredicate { ty, bounds });

            if self.at(TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(predicates)
    }

    // ==================== TYPES ====================

    fn parse_type(&mut self) -> Result<TypeExpr> {
        self.parse_type_with_precedence(0)
    }

    fn parse_type_with_precedence(&mut self, min_prec: u8) -> Result<TypeExpr> {
        let mut left = self.parse_type_primary()?;

        // Handle function types: A -> B
        while self.at(TokenKind::Arrow) && min_prec == 0 {
            self.advance();
            let ret = self.parse_type_with_precedence(1)?;
            left = TypeExpr::Function {
                params: vec![left],
                return_type: Box::new(ret),
                effects: Vec::new(),
            };
        }

        Ok(left)
    }

    fn parse_type_primary(&mut self) -> Result<TypeExpr> {
        match self.peek() {
            // Reference types
            TokenKind::Amp => {
                self.advance();
                let is_mut = if self.at(TokenKind::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };
                let inner = self.parse_type_primary()?;
                Ok(TypeExpr::Reference {
                    mutable: is_mut,
                    inner: Box::new(inner),
                })
            }

            // Array/slice types
            TokenKind::LBracket => {
                self.advance();
                let element = self.parse_type()?;

                if self.at(TokenKind::Semi) {
                    // Fixed-size array: [T; N]
                    self.advance();
                    let size = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;
                    Ok(TypeExpr::Array {
                        element: Box::new(element),
                        size: Some(Box::new(size)),
                    })
                } else {
                    // Slice: [T]
                    self.expect(TokenKind::RBracket)?;
                    Ok(TypeExpr::Array {
                        element: Box::new(element),
                        size: None,
                    })
                }
            }

            // Tuple types
            TokenKind::LParen => {
                self.advance();
                if self.at(TokenKind::RParen) {
                    self.advance();
                    return Ok(TypeExpr::Unit);
                }

                let mut elements = vec![self.parse_type()?];
                while self.at(TokenKind::Comma) {
                    self.advance();
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    elements.push(self.parse_type()?);
                }
                self.expect(TokenKind::RParen)?;

                if elements.len() == 1 {
                    // Single element with trailing comma is a tuple
                    Ok(TypeExpr::Tuple(elements))
                } else {
                    Ok(TypeExpr::Tuple(elements))
                }
            }

            // Named type
            TokenKind::Ident => {
                let path = self.parse_path()?;
                let args = if self.at(TokenKind::Lt) {
                    self.parse_type_args()?
                } else {
                    Vec::new()
                };

                // Check for unit suffix
                let unit = if self.at(TokenKind::Lt) && self.peek_n(1) != TokenKind::Ident {
                    // Already parsed type args
                    None
                } else {
                    None
                };

                Ok(TypeExpr::Named { path, args, unit })
            }

            // Infer type
            TokenKind::Underscore => {
                self.advance();
                Ok(TypeExpr::Infer)
            }

            _ => Err(miette::miette!("Expected type, found {:?}", self.peek())),
        }
    }

    // ==================== EXPRESSIONS ====================

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_expr_with_precedence(0)
    }

    /// Parse an expression without allowing struct literals
    /// Used in contexts like match scrutinee where `x { ... }` is ambiguous
    fn parse_expr_no_struct(&mut self) -> Result<Expr> {
        let old = self.allow_struct_literals;
        self.allow_struct_literals = false;
        let result = self.parse_expr();
        self.allow_struct_literals = old;
        result
    }

    fn parse_expr_with_precedence(&mut self, min_prec: u8) -> Result<Expr> {
        let mut left = self.parse_unary()?;

        while let Some((op, prec, assoc)) = self.binary_op_info() {
            if prec < min_prec {
                break;
            }

            self.advance();
            let next_min = if assoc == Assoc::Left { prec + 1 } else { prec };
            let right = self.parse_expr_with_precedence(next_min)?;

            left = Expr::Binary {
                id: self.next_id(),
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn binary_op_info(&self) -> Option<(BinaryOp, u8, Assoc)> {
        let (op, prec, assoc) = match self.peek() {
            TokenKind::PipePipe => (BinaryOp::Or, 1, Assoc::Left),
            TokenKind::AmpAmp => (BinaryOp::And, 2, Assoc::Left),
            TokenKind::EqEq => (BinaryOp::Eq, 3, Assoc::Left),
            TokenKind::Ne => (BinaryOp::Ne, 3, Assoc::Left),
            TokenKind::Lt => (BinaryOp::Lt, 4, Assoc::Left),
            TokenKind::Le => (BinaryOp::Le, 4, Assoc::Left),
            TokenKind::Gt => (BinaryOp::Gt, 4, Assoc::Left),
            TokenKind::Ge => (BinaryOp::Ge, 4, Assoc::Left),
            TokenKind::Pipe => (BinaryOp::BitOr, 5, Assoc::Left),
            TokenKind::Caret => (BinaryOp::BitXor, 6, Assoc::Left),
            TokenKind::Amp => (BinaryOp::BitAnd, 7, Assoc::Left),
            TokenKind::Shl => (BinaryOp::Shl, 8, Assoc::Left),
            TokenKind::Shr => (BinaryOp::Shr, 8, Assoc::Left),
            TokenKind::Plus => (BinaryOp::Add, 9, Assoc::Left),
            TokenKind::Minus => (BinaryOp::Sub, 9, Assoc::Left),
            TokenKind::Star => (BinaryOp::Mul, 10, Assoc::Left),
            TokenKind::Slash => (BinaryOp::Div, 10, Assoc::Left),
            TokenKind::Percent => (BinaryOp::Rem, 10, Assoc::Left),
            _ => return None,
        };
        Some((op, prec, assoc))
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            TokenKind::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            TokenKind::Bang => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            TokenKind::Amp => {
                self.advance();
                let is_mut = if self.at(TokenKind::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: if is_mut {
                        UnaryOp::RefMut
                    } else {
                        UnaryOp::Ref
                    },
                    expr: Box::new(expr),
                })
            }
            TokenKind::Star => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: UnaryOp::Deref,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek() {
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(TokenKind::RParen) {
                        args.push(self.parse_expr()?);
                        if !self.at(TokenKind::RParen) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    expr = Expr::Call {
                        id: self.next_id(),
                        callee: Box::new(expr),
                        args,
                    };
                }
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;
                    expr = Expr::Index {
                        id: self.next_id(),
                        base: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    if self.at(TokenKind::IntLit) {
                        // Tuple field access
                        let index: usize = self.advance().text.parse().unwrap_or(0);
                        expr = Expr::TupleField {
                            id: self.next_id(),
                            base: Box::new(expr),
                            index,
                        };
                    } else {
                        let field = self.parse_ident()?;
                        expr = Expr::Field {
                            id: self.next_id(),
                            base: Box::new(expr),
                            field,
                        };
                    }
                }
                TokenKind::Question => {
                    self.advance();
                    expr = Expr::Try {
                        id: self.next_id(),
                        expr: Box::new(expr),
                    };
                }
                TokenKind::As => {
                    self.advance();
                    let ty = self.parse_type()?;
                    expr = Expr::Cast {
                        id: self.next_id(),
                        expr: Box::new(expr),
                        ty,
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek() {
            // Literals
            TokenKind::IntLit => {
                let text = self.advance().text.clone();
                let value: i64 = text.replace('_', "").parse().unwrap_or(0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::FloatLit => {
                let text = self.advance().text.clone();
                let value: f64 = text.replace('_', "").parse().unwrap_or(0.0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Float(value),
                })
            }
            TokenKind::StringLit => {
                let text = self.advance().text.clone();
                // Remove quotes
                let value = text[1..text.len() - 1].to_string();
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::String(value),
                })
            }
            TokenKind::CharLit => {
                let text = self.advance().text.clone();
                let value = text.chars().nth(1).unwrap_or('\0');
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Char(value),
                })
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Bool(true),
                })
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Bool(false),
                })
            }

            // Identifiers and paths
            TokenKind::Ident | TokenKind::SelfLower => {
                let path = self.parse_path()?;

                // Check for struct literal (only if allowed in this context)
                if self.allow_struct_literals
                    && self.at(TokenKind::LBrace)
                    && !path.segments.is_empty()
                {
                    return self.parse_struct_literal(path);
                }

                Ok(Expr::Path {
                    id: self.next_id(),
                    path,
                })
            }

            // Grouped expression or tuple
            TokenKind::LParen => {
                self.advance();
                if self.at(TokenKind::RParen) {
                    self.advance();
                    return Ok(Expr::Literal {
                        id: self.next_id(),
                        value: Literal::Unit,
                    });
                }

                let expr = self.parse_expr()?;

                if self.at(TokenKind::Comma) {
                    // Tuple
                    let mut elements = vec![expr];
                    while self.at(TokenKind::Comma) {
                        self.advance();
                        if self.at(TokenKind::RParen) {
                            break;
                        }
                        elements.push(self.parse_expr()?);
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Tuple {
                        id: self.next_id(),
                        elements,
                    })
                } else {
                    self.expect(TokenKind::RParen)?;
                    Ok(expr)
                }
            }

            // Array literal
            TokenKind::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                while !self.at(TokenKind::RBracket) {
                    elements.push(self.parse_expr()?);
                    if !self.at(TokenKind::RBracket) {
                        self.expect(TokenKind::Comma)?;
                    }
                }
                self.expect(TokenKind::RBracket)?;
                Ok(Expr::Array {
                    id: self.next_id(),
                    elements,
                })
            }

            // Block expression
            TokenKind::LBrace => {
                let block = self.parse_block()?;
                Ok(Expr::Block {
                    id: self.next_id(),
                    block,
                })
            }

            // If expression
            TokenKind::If => self.parse_if(),

            // Match expression
            TokenKind::Match => self.parse_match(),

            // Loop expressions
            TokenKind::Loop => self.parse_loop(),
            TokenKind::While => self.parse_while(),
            TokenKind::For => self.parse_for(),

            // Return
            TokenKind::Return => {
                self.advance();
                let value = if self.at_any(&[TokenKind::RBrace, TokenKind::Semi, TokenKind::Eof]) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Return {
                    id: self.next_id(),
                    value,
                })
            }

            // Break
            TokenKind::Break => {
                self.advance();
                let value = if self.at_any(&[TokenKind::RBrace, TokenKind::Semi, TokenKind::Eof]) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Break {
                    id: self.next_id(),
                    value,
                })
            }

            // Continue
            TokenKind::Continue => {
                self.advance();
                Ok(Expr::Continue { id: self.next_id() })
            }

            // Closure
            TokenKind::Pipe => self.parse_closure(),

            // Effect operations
            TokenKind::Perform => {
                self.advance();
                let effect = self.parse_path()?;
                self.expect(TokenKind::Dot)?;
                let op = self.parse_ident()?;
                let args = if self.at(TokenKind::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(TokenKind::RParen) {
                        args.push(self.parse_expr()?);
                        if !self.at(TokenKind::RParen) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    args
                } else {
                    Vec::new()
                };
                Ok(Expr::Perform {
                    id: self.next_id(),
                    effect,
                    op,
                    args,
                })
            }

            TokenKind::Handle => {
                self.advance();
                let expr = Box::new(self.parse_expr()?);
                self.expect(TokenKind::With)?;
                let handler = self.parse_path()?;
                Ok(Expr::Handle {
                    id: self.next_id(),
                    expr,
                    handler,
                })
            }

            // Probabilistic operations
            TokenKind::Sample => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let dist = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(Expr::Sample {
                    id: self.next_id(),
                    distribution: Box::new(dist),
                })
            }

            _ => Err(miette::miette!(
                "Unexpected token {:?} in expression",
                self.peek()
            )),
        }
    }

    fn parse_struct_literal(&mut self, path: Path) -> Result<Expr> {
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while !self.at(TokenKind::RBrace) {
            let name = self.parse_ident()?;
            let value = if self.at(TokenKind::Colon) {
                self.advance();
                self.parse_expr()?
            } else {
                // Shorthand: name without : means name: name
                Expr::Path {
                    id: self.next_id(),
                    path: Path {
                        segments: vec![name.clone()],
                    },
                }
            };
            fields.push((name, value));

            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr::StructLit {
            id: self.next_id(),
            path,
            fields,
        })
    }

    fn parse_if(&mut self) -> Result<Expr> {
        self.expect(TokenKind::If)?;
        let condition = Box::new(self.parse_expr()?);
        let then_branch = self.parse_block()?;
        let else_branch = if self.at(TokenKind::Else) {
            self.advance();
            if self.at(TokenKind::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                Some(Box::new(Expr::Block {
                    id: self.next_id(),
                    block: self.parse_block()?,
                }))
            }
        } else {
            None
        };

        Ok(Expr::If {
            id: self.next_id(),
            condition,
            then_branch,
            else_branch,
        })
    }

    fn parse_match(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Match)?;
        // Use parse_expr_no_struct to avoid ambiguity with `match x { ... }`
        // being parsed as struct literal `x { ... }`
        let scrutinee = Box::new(self.parse_expr_no_struct()?);
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let pattern = self.parse_pattern()?;
            let guard = if self.at(TokenKind::If) {
                self.advance();
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expr()?;
            if self.at(TokenKind::Comma) {
                self.advance();
            }
            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr::Match {
            id: self.next_id(),
            scrutinee,
            arms,
        })
    }

    fn parse_loop(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Loop)?;
        let body = self.parse_block()?;
        Ok(Expr::Loop {
            id: self.next_id(),
            body,
        })
    }

    fn parse_while(&mut self) -> Result<Expr> {
        self.expect(TokenKind::While)?;
        let condition = Box::new(self.parse_expr()?);
        let body = self.parse_block()?;
        Ok(Expr::While {
            id: self.next_id(),
            condition,
            body,
        })
    }

    fn parse_for(&mut self) -> Result<Expr> {
        self.expect(TokenKind::For)?;
        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::In)?;
        let iter = Box::new(self.parse_expr()?);
        let body = self.parse_block()?;
        Ok(Expr::For {
            id: self.next_id(),
            pattern,
            iter,
            body,
        })
    }

    fn parse_closure(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Pipe)?;
        let mut params = Vec::new();
        while !self.at(TokenKind::Pipe) {
            let name = self.parse_ident()?;
            let ty = if self.at(TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push((name, ty));
            if !self.at(TokenKind::Pipe) {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(TokenKind::Pipe)?;

        let return_type = if self.at(TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.at(TokenKind::LBrace) {
            let block = self.parse_block()?;
            Box::new(Expr::Block {
                id: self.next_id(),
                block,
            })
        } else {
            Box::new(self.parse_expr()?)
        };

        Ok(Expr::Closure {
            id: self.next_id(),
            params,
            return_type,
            body,
        })
    }

    // ==================== STATEMENTS ====================

    fn parse_block(&mut self) -> Result<Block> {
        self.expect(TokenKind::LBrace)?;
        let mut stmts = Vec::new();

        while !self.at(TokenKind::RBrace) {
            stmts.push(self.parse_stmt()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Block { stmts })
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        match self.peek() {
            TokenKind::Let => self.parse_let_stmt(),
            TokenKind::Semi => {
                self.advance();
                Ok(Stmt::Empty)
            }
            _ => {
                let expr = self.parse_expr()?;

                // Check for assignment
                if let Some(op) = self.assignment_op() {
                    self.advance();
                    let rhs = self.parse_expr()?;
                    if self.at(TokenKind::Semi) {
                        self.advance();
                    }
                    return Ok(Stmt::Assign {
                        target: expr,
                        op,
                        value: rhs,
                    });
                }

                let has_semi = if self.at(TokenKind::Semi) {
                    self.advance();
                    true
                } else {
                    false
                };

                Ok(Stmt::Expr { expr, has_semi })
            }
        }
    }

    fn assignment_op(&self) -> Option<AssignOp> {
        match self.peek() {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::PlusEq => Some(AssignOp::AddAssign),
            TokenKind::MinusEq => Some(AssignOp::SubAssign),
            TokenKind::StarEq => Some(AssignOp::MulAssign),
            TokenKind::SlashEq => Some(AssignOp::DivAssign),
            TokenKind::PercentEq => Some(AssignOp::RemAssign),
            TokenKind::AmpEq => Some(AssignOp::BitAndAssign),
            TokenKind::PipeEq => Some(AssignOp::BitOrAssign),
            TokenKind::CaretEq => Some(AssignOp::BitXorAssign),
            TokenKind::ShlEq => Some(AssignOp::ShlAssign),
            TokenKind::ShrEq => Some(AssignOp::ShrAssign),
            _ => None,
        }
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenKind::Let)?;
        let is_mut = if self.at(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };

        let pattern = self.parse_pattern()?;
        let ty = if self.at(TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let value = if self.at(TokenKind::Eq) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        if self.at(TokenKind::Semi) {
            self.advance();
        }

        Ok(Stmt::Let {
            is_mut,
            pattern,
            ty,
            value,
        })
    }

    // ==================== PATTERNS ====================

    fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.peek() {
            TokenKind::Underscore => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::IntLit => {
                let text = self.advance().text.clone();
                let value: i64 = text.replace('_', "").parse().unwrap_or(0);
                Ok(Pattern::Literal(Literal::Int(value)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::StringLit => {
                let text = self.advance().text.clone();
                let value = text[1..text.len() - 1].to_string();
                Ok(Pattern::Literal(Literal::String(value)))
            }
            TokenKind::LParen => {
                self.advance();
                if self.at(TokenKind::RParen) {
                    self.advance();
                    return Ok(Pattern::Literal(Literal::Unit));
                }
                let mut elements = vec![self.parse_pattern()?];
                while self.at(TokenKind::Comma) {
                    self.advance();
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    elements.push(self.parse_pattern()?);
                }
                self.expect(TokenKind::RParen)?;
                Ok(Pattern::Tuple(elements))
            }
            TokenKind::Ident | TokenKind::SelfLower => {
                let path = self.parse_path()?;
                if self.at(TokenKind::LParen) {
                    // Enum variant with tuple data
                    self.advance();
                    let mut patterns = Vec::new();
                    while !self.at(TokenKind::RParen) {
                        patterns.push(self.parse_pattern()?);
                        if !self.at(TokenKind::RParen) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Pattern::Enum {
                        path,
                        patterns: Some(patterns),
                    })
                } else if self.at(TokenKind::LBrace) {
                    // Struct pattern
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) {
                        let name = self.parse_ident()?;
                        let pattern = if self.at(TokenKind::Colon) {
                            self.advance();
                            self.parse_pattern()?
                        } else {
                            Pattern::Binding {
                                name: name.clone(),
                                mutable: false,
                            }
                        };
                        fields.push((name, pattern));
                        if !self.at(TokenKind::RBrace) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RBrace)?;
                    Ok(Pattern::Struct { path, fields })
                } else if path.segments.len() == 1 {
                    // Simple binding
                    Ok(Pattern::Binding {
                        name: path.segments.into_iter().next().unwrap(),
                        mutable: false,
                    })
                } else {
                    // Path pattern (unit variant)
                    Ok(Pattern::Enum {
                        path,
                        patterns: None,
                    })
                }
            }
            TokenKind::Mut => {
                self.advance();
                let name = self.parse_ident()?;
                Ok(Pattern::Binding {
                    name,
                    mutable: true,
                })
            }
            _ => Err(miette::miette!("Expected pattern, found {:?}", self.peek())),
        }
    }

    // ==================== HELPERS ====================

    fn parse_ident(&mut self) -> Result<String> {
        if self.at(TokenKind::Ident) {
            Ok(self.advance().text.clone())
        } else if self.at(TokenKind::SelfLower) {
            Ok(self.advance().text.clone())
        } else {
            Err(miette::miette!(
                "Expected identifier, found {:?}",
                self.peek()
            ))
        }
    }

    fn parse_path(&mut self) -> Result<Path> {
        let mut segments = vec![self.parse_ident()?];

        while self.at(TokenKind::ColonColon) {
            self.advance();
            segments.push(self.parse_ident()?);
        }

        Ok(Path { segments })
    }
}

/// Associativity for binary operators
#[derive(Clone, Copy, PartialEq, Eq)]
enum Assoc {
    Left,
    Right,
}
