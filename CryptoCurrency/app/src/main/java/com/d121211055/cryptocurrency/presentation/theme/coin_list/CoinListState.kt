package com.d121211055.cryptocurrency.presentation.theme.coin_list

import com.d121211055.cryptocurrency.domain.model.Coin

data class CoinListState(
    val isLoading: Boolean = false,
    val coins: List<Coin> = emptyList(),
    val error: String = ""
)
